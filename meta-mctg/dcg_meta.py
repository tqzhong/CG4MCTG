import torch
import sys
sys.path.append('..')
from data.utils import get_train_dataset
from transformers import GPT2Tokenizer, GPT2Config
from modeling_gpt2 import GPT2LMHeadModel
from transformers import get_linear_schedule_with_warmup, AdamW
import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
import argparse
import random
import numpy as np
from tqdm import tqdm, trange
import itertools
from collections import defaultdict, deque
import copy
import json
import math
import os

def tokenize(dataset_path:list, tokenizer) -> list:
    '''
    tokenize the data, the tokenized data is formulated as (take the dataset YELP as an example):
    [
        {
            "text": [], 
            "sentiment": tokenizer.encode(" positive"), 
            "pronoun": tokenizer.encode(" singular"), 
            "tense": tokenizer.encode(" past"), 
            "comb": {"sentiment": "positive", "pronoun": "singular", "tense": "past"}
        },
        ...
    ]
    '''
    tokenized_data = list()
    for dic in tqdm(dataset_path, desc='Tokenizing'):
        new_dic = {}
        new_dic['text'] = tokenizer.encode(dic['review'], max_length=512, truncation=True)
        attribute_keys = list(dic.keys())[:]
        attribute_keys.remove('review')
        new_dic['comb'] = dict()
        for key in attribute_keys:
            new_dic[key] = tokenizer.encode(' ' + dic[key])
            new_dic['comb'][key] = dic[key] 
        tokenized_data.append(new_dic)
    return tokenized_data

def padding_fuse_fn(data_list:list) -> dict:
    '''
    padding function
    '''
    input_ids = list()
    attention_masks = list()
    text_length = list()
    labels = dict()
    combs = list()

    key_list = list(data_list[0].keys())
    key_list.remove('text')
    for key in key_list:
        labels[key] = list()

    for item in data_list:
        text_length.append(len(item['text']))
        combs.append(item['comb'])
        for key in key_list:
            labels[key].append(item[key])
    max_text_len = max(text_length)
    for i, item in enumerate(data_list):
        text_pad_len = max_text_len - text_length[i]
        attention_mask = [1] * text_length[i] + [0] * text_pad_len
        text = item["text"] + [50256] * text_pad_len

        input_ids.append(text)
        attention_masks.append(attention_mask)
    
    batch = dict()
    batch['input_ids'] = input_ids
    batch['attention_mask'] = attention_masks
    batch['comb'] = combs
    for key in key_list:
        batch[key] = labels[key]

    return batch

class tokendataset(Dataset):
    def __init__(self, dataset_path):
        self.file_path = dataset_path
        file_row = len(dataset_path)

        self.file_row = file_row
        self.dataset = dataset_path

    def __len__(self):
        return self.file_row
    
    def __getitem__(self, idx):
        return self.dataset[idx]

def set_seed(seed:int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_att_tokens_ids(combs:list, tokenizer) -> list:
    '''
    encode attribute tokens
    '''
    res_att_tokens_ids = list()
    for item in combs:
        att_tokens_ids = list()
        for key in list(item.keys()):
            assert len(tokenizer.encode(' ' + item[key])) == 1
            att_tokens_ids.append(tokenizer.encode(' ' + item[key])[0])
        res_att_tokens_ids.append(att_tokens_ids)
    return res_att_tokens_ids

def get_support_combs(current_combs:list, seen_combs:list, label_keys:list) -> list:
    current_single_att = dict()
    for key in label_keys:
        current_single_att[key] = list()
    for comb in current_combs:
        for key in label_keys:
            current_single_att[key].append(comb[key])
    for key in label_keys:
        current_single_att[key] = [current_single_att[key][i] for i in range(len(current_single_att[key])) if current_single_att[key][i] not in current_single_att[key][:i]]
    
    keys = current_single_att.keys()
    values = current_single_att.values()
    combinations = list(itertools.product(*values))
    available_combs = [dict(zip(keys, combination)) for combination in combinations]

    support_combs = [item for item in available_combs if item not in current_combs]
    support_combs = [item for item in support_combs if item in seen_combs]

    return support_combs

def get_support_batch(support_combs:list, args) -> list:
    support_data = list()
    support_num = args.batch_size * args.gradient_accumulation_steps
    all_support_att_tokens_ids = get_att_tokens_ids(combs=support_combs, tokenizer=args.tokenizer)
    if args.num_pseu >= len(all_support_att_tokens_ids):
        args.support_num_pseu = len(all_support_att_tokens_ids)
    else:
        args.support_num_pseu = args.num_pseu
    
    if support_num <= len(support_combs):
        sample_combs = random.sample(support_combs, support_num)
        for comb in sample_combs:
            data_list = args.label_to_tokenized_data[json.dumps(comb)]
            sample_data = random.choice(data_list)
            support_data.append(sample_data)
    else:
        support_num_per_comb = int(support_num / len(support_combs))
        for comb in support_combs:
            data_list = args.label_to_tokenized_data[json.dumps(comb)]
            sample_data = random.sample(data_list, support_num_per_comb)
            support_data.extend(sample_data)
        residual_support_num = support_num - support_num_per_comb * len(support_combs)
        if residual_support_num != 0:
            data_lists = [args.label_to_tokenized_data[json.dumps(comb)] for comb in support_combs]
            combined_data_list = [item for sublist in data_lists for item in sublist]
            for sampled_data in support_data:
                combined_data_list.remove(sampled_data)
            support_data.extend(random.sample(combined_data_list, residual_support_num))
    assert len(support_data) == support_num
    return support_data

def train(args):
    '''
    training phase
    '''
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(args.seed)

    config = GPT2Config.from_pretrained(args.model_name_or_path)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    args.tokenizer = tokenizer

    tokenized_data = tokenize(dataset_path=args.train_dataset, tokenizer=args.tokenizer)
    train_dataset = tokendataset(tokenized_data)
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=False, collate_fn=padding_fuse_fn, sampler=train_sampler)

    # store data by labels (attribute combination)
    label_to_tokenized_data = defaultdict(deque)
    for item in tokenized_data:
        label_to_tokenized_data[json.dumps(item['comb'])].append(item)
    args.label_to_tokenized_data = label_to_tokenized_data
    
    seen_att_tokens_ids = get_att_tokens_ids(combs=args.seen_combs, tokenizer=tokenizer)
    label_keys = list(args.all_combs[0].keys())
    if args.num_pseu >= len(seen_att_tokens_ids):
        args.num_pseu = len(seen_att_tokens_ids)
    
    # set the config
    config.is_dcg = True
    config.dcg_att_num = len(label_keys)
    config.dcg_att_len = args.dcg_att_len
    config.dcg_task_len = args.dcg_task_len
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path, config=config)

    # frozen parameters
    for param in model.named_parameters():
        if 'dcg' in param[0]:
            continue
        else:
            param[1].requires_grad = False

    model.to(args.device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    num_train_steps = math.floor(len(train_dataset) / (
            args.batch_size * args.gradient_accumulation_steps)) * args.num_train_epochs
    num_warmup_steps = math.floor(num_train_steps * args.warmup_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    tr_loss_lm, logging_loss_lm = 0.0, 0.0
    tr_loss_dis, logging_loss_dis = 0.0, 0.0
    tr_sloss, logging_sloss = 0.0, 0.0
    tr_sloss_lm, logging_sloss_lm = 0.0, 0.0 
    tr_sloss_dis, logging_sloss_dis = 0.0, 0.0
    model.train()
    model.zero_grad()
    loss_fct = CrossEntropyLoss(ignore_index=50256)
    args.loss_fct = loss_fct
    logger.info('start_training')

    if args.meta_mctg:
        mini_batch = args.batch_size * args.gradient_accumulation_steps
        if len(args.seen_combs) < mini_batch:
            args.sample_train = True
    
    if not args.sample_train:
        '''
        number of seen_combs >= mini_batch. Training using random traversal of the training data
        '''

        if not args.meta_mctg:
            print("="*100)
            print("Apply Common Training")
            print("="*100)
        else:
            print("="*100)
            print("Apply Meta Training")
            print("="*100)

        current_epoch = 0
        for epoch in trange(int(args.num_train_epochs), desc='Epoch'):
            current_epoch += 1

            current_combs = list()
            for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):
                global_step += 1
                input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
                label_ids = dict()

                for key in label_keys:
                    label_ids[key] = torch.tensor(batch[key])
                
                combs_step = batch['comb']
                current_combs.extend(combs_step)

                att_tokens_ids = None
                for key in label_keys:
                    if att_tokens_ids is None:
                        att_tokens_ids = label_ids[key]
                    else:
                        att_tokens_ids = torch.cat([att_tokens_ids, label_ids[key]], dim=-1)
                att_tokens_ids = att_tokens_ids.to(args.device)
                    
                eos_token_ids = torch.tensor(tokenizer.encode(tokenizer.eos_token))
                eos_token_ids = eos_token_ids.expand(args.batch_size, eos_token_ids.shape[0]).to(args.device)
                input_ids = torch.tensor(input_ids).to(args.device)
                input_ids = torch.cat([eos_token_ids, input_ids], dim=-1)

                prompt_len = args.dcg_att_len + args.dcg_task_len
                eos_token_mask = torch.tensor([1]).expand(args.batch_size, 1).to(args.device)
                prompt_mask = torch.tensor([1] * prompt_len).expand(args.batch_size, prompt_len).to(args.device)
                attention_mask = torch.tensor(attention_mask).to(args.device)
                attention_mask = torch.cat([prompt_mask, attention_mask], dim=-1)
                attention_mask = torch.cat([eos_token_mask, attention_mask], dim=-1)

                dic = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, use_cache=True, config=config, att_tokens_ids=att_tokens_ids)
                logits = dic.logits
                shift_logits = logits[:, prompt_len:-1, :].contiguous()
                labels = input_ids[:, 1:].contiguous()
                loss_lm = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

                pseu_combinations_set = random.sample(seen_att_tokens_ids, args.num_pseu)
                loss_set = list()
                loss_set.append(torch.exp(-loss_lm))
                for pseu_set in pseu_combinations_set:
                    att_tokens_ids = torch.tensor(pseu_set).unsqueeze(0).expand(args.batch_size, len(pseu_set)).to(args.device)
                    dic = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, use_cache=True, config=config, att_tokens_ids=att_tokens_ids)
                    logits = dic.logits
                    shift_logits = logits[:, prompt_len:-1, :].contiguous()
                    labels = input_ids[:, 1:].contiguous()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
                    loss_set.append(torch.exp(-loss))
                    
                loss_dis = loss_lm + torch.log(sum(loss_set))
                loss = args.alpha * loss_dis + (1 - args.alpha) * loss_lm

                loss.backward()
                tr_loss += loss.item()
                tr_loss_lm += loss_lm.item()
                tr_loss_dis += loss_dis.item()

                if (step + 1 ) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    if args.meta_mctg is not True:
                        # common dcg training
                        optimizer.step()
                        scheduler.step()
                        model.zero_grad()
                    else:
                        # use meta_mctg training
                        current_combs = [current_combs[i] for i in range(len(current_combs)) if current_combs[i] not in current_combs[:i]]
                        # get pseudo-comp attribute combinations set
                        support_combs = get_support_combs(current_combs=current_combs, seen_combs=args.seen_combs, label_keys=label_keys)
                        if len(support_combs) == 0:
                            pass
                        else:
                            # get pseudo-comp batch
                            support_data = get_support_batch(support_combs=support_combs, args=args)

                            all_support_att_tokens_ids = get_att_tokens_ids(combs=support_combs, tokenizer=tokenizer)

                            support_sampler = torch.utils.data.RandomSampler(support_data)
                            support_dataloader = DataLoader(tokendataset(support_data), batch_size=args.batch_size, drop_last=False, collate_fn=padding_fuse_fn, sampler=support_sampler)
                            # create a backup model and store the parameter after training a train batch
                            backup_model = copy.deepcopy(model)
                            backup_model.to(args.device)
                            with torch.no_grad():
                                for param, backup_param in zip(model.parameters(), backup_model.parameters()):
                                    if param.grad is not None:
                                        backup_param -= optimizer.param_groups[0]['lr'] * param.grad
                            for support_batch in support_dataloader:
                                support_input_ids, support_attention_mask = support_batch['input_ids'], support_batch['attention_mask']
                                support_label_ids = dict()
                                for key in label_keys:
                                    support_label_ids[key] = torch.tensor(support_batch[key])
                                
                                support_att_tokens_ids = None
                                for key in label_keys:
                                    if support_att_tokens_ids is None:
                                        support_att_tokens_ids = support_label_ids[key]
                                    else:
                                        support_att_tokens_ids = torch.cat([support_att_tokens_ids, support_label_ids[key]], dim=-1)
                                support_att_tokens_ids = support_att_tokens_ids.to(args.device)

                                support_input_ids = torch.tensor(support_input_ids).to(args.device)
                                support_input_ids = torch.cat([eos_token_ids, support_input_ids], dim=-1)
                                support_attention_mask = torch.tensor(support_attention_mask).to(args.device)
                                support_attention_mask = torch.cat([prompt_mask, support_attention_mask], dim=-1)
                                support_attention_mask = torch.cat([eos_token_mask, support_attention_mask], dim=-1)

                                support_dic = backup_model(input_ids=support_input_ids, attention_mask=support_attention_mask, return_dict=True, use_cache=True, config=config, att_tokens_ids=support_att_tokens_ids)
                                support_logits = support_dic.logits
                                support_shift_logits = support_logits[:, prompt_len:-1, :].contiguous()
                                support_labels = support_input_ids[:, 1:].contiguous()
                                loss_support_lm = loss_fct(support_shift_logits.view(-1, support_shift_logits.size(-1)), support_labels.view(-1))

                                support_pseu_combinations_set = random.sample(all_support_att_tokens_ids, args.support_num_pseu)
                                s_loss_set = list()
                                s_loss_set.append(torch.exp(-loss_support_lm))
                                for support_pseu_set in support_pseu_combinations_set:
                                    support_att_tokens_ids = torch.tensor(support_pseu_set).unsqueeze(0).expand(args.batch_size, len(support_pseu_set)).to(args.device)
                                    support_dic = backup_model(input_ids=support_input_ids, attention_mask=support_attention_mask, return_dict=True, use_cache=True, config=config, att_tokens_ids=support_att_tokens_ids)
                                    support_logits = support_dic.logits
                                    support_shift_logits = support_logits[:, prompt_len:-1, :].contiguous()
                                    s_loss = loss_fct(support_shift_logits.view(-1, support_shift_logits.size(-1)), support_labels.view(-1))
                                    s_loss_set.append(torch.exp(-s_loss))
                                
                                loss_support_dis = loss_support_lm + torch.log(sum(s_loss_set))

                                loss_support = args.lambda_s * (args.alpha * loss_support_dis + (1 - args.alpha) * loss_support_lm)
                                loss_support.backward()

                                tr_sloss += loss_support.item()
                                tr_sloss_lm += loss_support_lm.item()
                                tr_sloss_dis += loss_support_dis.item()
                            
                            # add the gradient of pseudo-comp batch to original model
                            for param, backup_param in zip(model.parameters(), backup_model.parameters()):
                                if backup_param.grad is not None:
                                    if param.grad is not None:
                                        param.grad += backup_param.grad
                                    else:
                                        param.grad = backup_param.grad.clone()
                        
                        # renew the current combinations list
                        current_combs = list()

                        # truly update
                        optimizer.step()
                        scheduler.step()
                        model.zero_grad()
                    
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    loss_lm_scalar = (tr_loss_lm - logging_loss_lm) / args.logging_steps
                    loss_dis_scalar = (tr_loss_dis - logging_loss_dis) / args.logging_steps
                    if args.meta_mctg:
                        loss_s_scalar = (tr_sloss - logging_sloss) / args.logging_steps
                        loss_s_lm_scalar = (tr_sloss_lm - logging_sloss_lm) / args.logging_steps
                        loss_s_dis_scalar = (tr_sloss_dis - logging_sloss_dis) / args.logging_steps
                    logs['epoch'] = current_epoch
                    logs['step'] = global_step
                    logs['loss'] = loss_scalar
                    logs['lm_loss'] = loss_lm_scalar
                    logs['dis_loss'] = loss_dis_scalar
                    if args.meta_mctg:
                        logs['sloss'] = loss_s_scalar
                        logs['lm_sloss'] = loss_s_lm_scalar
                        logs['dis_sloss'] = loss_s_dis_scalar
                    logs['lr'] = optimizer.param_groups[0]['lr']
                    logging_loss = tr_loss
                    logging_loss_lm = tr_loss_lm
                    logging_loss_dis = tr_loss_dis
                    if args.meta_mctg:
                        logging_sloss = tr_sloss
                        logging_sloss_lm = tr_sloss_lm
                        logging_sloss_dis = tr_sloss_dis
                    print(logs)
                
            if current_epoch <= args.num_train_epochs:
                if not args.meta_mctg:
                    args.epoch_name = 'dcg-{}-{}-bs={}-epoch={}'.format(args.dataset, args.mode_name, args.batch_size * args.gradient_accumulation_steps, current_epoch)
                else:
                    args.epoch_name = 'dcg_meta-{}-{}-lambda={}-bs={}-epoch={}'.format(args.dataset, args.mode_name, args.lambda_s, args.batch_size * args.gradient_accumulation_steps, current_epoch)

                output_dir = os.path.join(args.output_dir, args.epoch_name)
                model_to_save = (model.module if hasattr(model, 'module') else model)
                model_to_save.save_pretrained(output_dir)
                config.save_pretrained(output_dir)

                args.finetuned_model = output_dir

                # generate phase
                test(args)
    else:
        '''
        number of seen combs < mini_batch. 
        Training by first sampling combinations (num of sampling combinations < num of seen combs) and then sampling data based on those combinations, ensuring that each epoch of training data approximates the result of traversing all the data. This approach helps to ensure the successful construction of pseudo-comp batch when number of seen combs < mini_batch.
        '''
        print("="*100)
        print("Applying Sample Meta Training")
        print("="*100)
        assert len(args.seen_combs) >= 2
        assert args.num_sample_combs is not None
        assert mini_batch % args.num_sample_combs == 0
        num_sample_combs = args.num_sample_combs
        new_gradient_accumulation_steps = int(mini_batch / num_sample_combs)
        new_batch_size = num_sample_combs
        
        current_epoch = 0
        for epoch in trange(int(args.num_train_epochs), desc='Epoch'): 
            current_epoch += 1
            backup_label_to_tokenized_data = copy.deepcopy(label_to_tokenized_data)
            num_combs_of_sample_combs = math.comb(len(args.seen_combs), num_sample_combs)
            samples_per_comb_of_sample_combs = int(len(tokenized_data) / num_combs_of_sample_combs / num_sample_combs)
            combs_of_comb_list = list()
            all_combs_of_comb = list(itertools.combinations(args.seen_combs, num_sample_combs))
            for item in all_combs_of_comb:
                for i in range(samples_per_comb_of_sample_combs):
                    combs_of_comb_list.append(item)
            # we ignore the residual data in the training set
            residual_num_per_comb = int((len(tokenized_data) - samples_per_comb_of_sample_combs * num_combs_of_sample_combs * num_sample_combs) / len(args.seen_combs))    

            random.shuffle(combs_of_comb_list)
            assert samples_per_comb_of_sample_combs >= new_gradient_accumulation_steps

            for comb_of_comb in tqdm(combs_of_comb_list, desc='Iteration'):
                if combs_of_comb_list.count(comb_of_comb) < new_gradient_accumulation_steps:
                    # current num of comb_of_comb can not support for a minibatch
                    continue
                current_combs = list()
                for i in range(new_gradient_accumulation_steps):
                    global_step += 1
                    combs_of_comb_list.remove(comb_of_comb)

                    batch_data = list()
                    for comb in comb_of_comb:
                        _sample_data = random.choice(backup_label_to_tokenized_data[json.dumps(comb)])
                        backup_label_to_tokenized_data[json.dumps(comb)].remove(_sample_data)
                        batch_data.append(_sample_data)
                    batch_data = padding_fuse_fn(batch_data)
                    input_ids, attention_mask = batch_data['input_ids'], batch_data['attention_mask']
                    label_ids = dict()

                    for key in label_keys:
                        label_ids[key] = torch.tensor(batch_data[key])
                    
                    combs_step = batch_data['comb']
                    current_combs.extend(combs_step)

                    att_tokens_ids = None
                    for key in label_keys:
                        if att_tokens_ids is None:
                            att_tokens_ids = label_ids[key]
                        else:
                            att_tokens_ids = torch.cat([att_tokens_ids, label_ids[key]], dim=1)
                    att_tokens_ids = att_tokens_ids.to(args.device)

                    eos_token_ids = torch.tensor(tokenizer.encode(tokenizer.eos_token))
                    eos_token_ids = eos_token_ids.expand(new_batch_size, eos_token_ids.shape[0]).to(args.device)
                    input_ids = torch.tensor(input_ids).to(args.device)
                    input_ids = torch.cat([eos_token_ids, input_ids], dim=-1)

                    prompt_len = args.dcg_att_len + args.dcg_task_len
                    eos_token_mask = torch.tensor([1]).expand(new_batch_size, 1).to(args.device)
                    prompt_mask = torch.tensor([1] * prompt_len).expand(new_batch_size, prompt_len).to(args.device)
                    attention_mask = torch.tensor(attention_mask).to(args.device)
                    attention_mask = torch.cat([prompt_mask, attention_mask], dim=-1)
                    attention_mask = torch.cat([eos_token_mask, attention_mask], dim=-1)

                    dic = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, use_cache=True, config=config, att_tokens_ids=att_tokens_ids)
                    logits = dic.logits
                    shift_logits = logits[:, prompt_len:-1, :].contiguous()
                    labels = input_ids[:, 1:].contiguous()
                    loss_lm = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
                    
                    pseu_combinations_set = random.sample(seen_att_tokens_ids, args.num_pseu)
                    loss_set = list()
                    loss_set.append(torch.exp(-loss_lm))
                    for pseu_set in pseu_combinations_set:
                        att_tokens_ids = torch.tensor(pseu_set).unsqueeze(0).expand(new_batch_size, len(pseu_set)).to(args.device)
                        dic = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, use_cache=True, config=config, att_tokens_ids=att_tokens_ids)
                        logits = dic.logits
                        shift_logits = logits[:, prompt_len:-1, :].contiguous()
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
                        loss_set.append(torch.exp(-loss))
                    
                    loss_dis = loss_lm + torch.log(sum(loss_set))
                    loss = args.alpha * loss_dis + (1 - args.alpha) * loss_lm

                    loss.backward()
                    tr_loss += loss.item()
                    tr_loss_lm += loss_lm.item()
                    tr_loss_dis += loss_dis.item()

                    if i == new_gradient_accumulation_steps - 1:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                        backup_model = copy.deepcopy(model)
                        backup_model.to(args.device)
                        with torch.no_grad():
                            for param, backup_param in zip(model.parameters(), backup_model.parameters()):
                                if param.grad is not None:
                                    backup_param -= optimizer.param_groups[0]['lr'] * param.grad

                        current_combs = [current_combs[i] for i in range(len(current_combs)) if current_combs[i] not in current_combs[:i]]
                        support_combs = get_support_combs(current_combs=current_combs, seen_combs=args.seen_combs, label_keys=label_keys)
                        all_support_att_tokens_ids = get_att_tokens_ids(combs=support_combs, tokenizer=tokenizer)
                        if len(support_combs) == 0:
                            pass
                        else:
                            support_data = get_support_batch(support_combs=support_combs, args=args)
                            support_sampler = torch.utils.data.RandomSampler(support_data)
                            support_dataloader = DataLoader(tokendataset(support_data), batch_size=args.batch_size, drop_last=False, collate_fn=padding_fuse_fn, sampler=support_sampler)

                            for support_batch in support_dataloader:
                                support_input_ids, support_attention_mask = support_batch['input_ids'], support_batch['attention_mask']
                                support_label_ids = dict()
                                for key in label_keys:
                                    support_label_ids[key] = torch.tensor(support_batch[key])
                                
                                support_att_tokens_ids = None
                                for key in label_keys:
                                    if support_att_tokens_ids is None:
                                        support_att_tokens_ids = support_label_ids[key]
                                    else:
                                        support_att_tokens_ids = torch.cat([support_att_tokens_ids, support_label_ids[key]], dim=-1)
                                support_att_tokens_ids = support_att_tokens_ids.to(args.device)

                                support_input_ids = torch.tensor(support_input_ids).to(args.device)
                                support_input_ids = torch.cat([eos_token_ids, support_input_ids], dim=-1)
                                support_attention_mask = torch.tensor(support_attention_mask).to(args.device)
                                support_attention_mask = torch.cat([prompt_mask, support_attention_mask], dim=-1)
                                support_attention_mask = torch.cat([eos_token_mask, support_attention_mask], dim=-1)

                                support_dic = backup_model(input_ids=support_input_ids, attention_mask=support_attention_mask, return_dict=True, use_cache=True, config=config, att_tokens_ids=support_att_tokens_ids)
                                support_logits = support_dic.logits
                                support_shift_logits = support_logits[:, prompt_len:-1, :].contiguous()
                                support_labels = support_input_ids[:, 1:].contiguous()
                                loss_support_lm = loss_fct(support_shift_logits.view(-1, support_shift_logits.size(-1)), support_labels.view(-1))

                                support_pseu_combinations_set = random.sample(all_support_att_tokens_ids, args.support_num_pseu)
                                s_loss_set = list()
                                s_loss_set.append(torch.exp(-loss_support_lm))
                                for support_pseu_set in support_pseu_combinations_set:
                                    support_att_tokens_ids = torch.tensor(support_pseu_set).unsqueeze(0).expand(args.batch_size, len(support_pseu_set)).to(args.device)
                                    support_dic = backup_model(input_ids=support_input_ids, attention_mask=support_attention_mask, return_dict=True, use_cache=True, config=config, att_tokens_ids=support_att_tokens_ids)
                                    support_logits = support_dic.logits
                                    support_shift_logits = support_logits[:, prompt_len:-1, :].contiguous()
                                    s_loss = loss_fct(support_shift_logits.view(-1, support_shift_logits.size(-1)), support_labels.view(-1))
                                    s_loss_set.append(torch.exp(-s_loss))
                                
                                loss_support_dis = loss_support_lm + torch.log(sum(s_loss_set))
                            
                                loss_support = args.lambda_s * (args.alpha * loss_support_dis + (1 - args.alpha) * loss_support_lm)
                                loss_support.backward()

                                tr_sloss += loss_support.item()
                                tr_sloss_lm += loss_support_lm.item()
                                tr_sloss_dis += loss_support_dis.item()
                        
                        for param, backup_param in zip(model.parameters(), backup_model.parameters()):
                            if backup_param.grad is not None:
                                if param.grad is not None:
                                    param.grad += backup_param.grad
                                else:
                                    param.grad = backup_param.grad.clone()
                    
                        current_combs = list()

                        optimizer.step()
                        scheduler.step()
                        model.zero_grad()

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                        loss_lm_scalar = (tr_loss_lm - logging_loss_lm) / args.logging_steps
                        loss_dis_scalar = (tr_loss_dis - logging_loss_dis) / args.logging_steps
                        loss_s_scalar = (tr_sloss - logging_sloss) / args.logging_steps
                        loss_s_lm_scalar = (tr_sloss_lm - logging_sloss_lm) / args.logging_steps
                        loss_s_dis_scalar = (tr_sloss_dis - logging_sloss_dis) / args.logging_steps
                        logs['epoch'] = current_epoch
                        logs['step'] = global_step
                        logs['loss'] = loss_scalar
                        logs['lm_loss'] = loss_lm_scalar
                        logs['dis_loss'] = loss_dis_scalar
                        logs['sloss'] = loss_s_scalar
                        logs['lm_sloss'] = loss_s_lm_scalar
                        logs['dis_sloss'] = loss_s_dis_scalar
                        logs['lr'] = optimizer.param_groups[0]['lr']
                        logging_loss = tr_loss
                        logging_loss_lm = tr_loss_lm
                        logging_loss_dis = tr_loss_dis
                        logging_sloss = tr_sloss
                        logging_sloss_lm = tr_sloss_lm
                        logging_sloss_dis = tr_sloss_dis
                        print(logs)
            
            if current_epoch <= args.num_train_epochs:
                args.epoch_name = 'dcg_sample_meta-{}-{}-lambda={}-{}-bs={}-epoch={}'.format(args.dataset_name, args.mode_name, args.lambda_s, args.num_sample_combs, args.batch_size * args.gradient_accumulation_steps, current_epoch)

                output_dir = os.path.join(args.output_dir, args.epoch_name)
                model_to_save = (model.module if hasattr(model, 'module') else model)
                model_to_save.save_pretrained(output_dir)
                config.save_pretrained(output_dir)

                args.finetuned_model = output_dir
                test(args)
    
    logger.info(' global_step = %s, average loss = %s', global_step, tr_loss / global_step)

def test(args):
    set_seed(args.test_seed)
    model = GPT2LMHeadModel.from_pretrained(args.finetuned_model).to(args.device)
    config = GPT2Config.from_pretrained(args.finetuned_model)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    args.tokenizer = tokenizer
    model.eval()

    '''
    construct all the possible combinations include seen_type (in the training dataset) and the unseen_type
    '''
    all_combs = args.all_combs
    seen_combs = args.seen_combs
    unseen_combs = args.unseen_combs

    file_seen = open(os.path.join(args.output_data_dir, "{}_seen.jsonl".format(args.epoch_name)), 'w')
    file_unseen = open(os.path.join(args.output_data_dir, "{}_unseen.jsonl".format(args.epoch_name)), 'w')

    for prompt in tqdm(args.prompt):
        for comb in all_combs:
            att_tokens_ids = list()
            for key in list(comb.keys()):
                assert len(tokenizer.encode(' ' + comb[key])) == 1
                att_tokens_ids.append(tokenizer.encode(' ' + comb[key])[0])
            att_tokens_ids = torch.tensor(att_tokens_ids).unsqueeze(0).expand(args.samples, len(att_tokens_ids)).to(args.device)
            with torch.no_grad():
                input_text = torch.tensor([tokenizer(tokenizer.eos_token + prompt).input_ids]).long().to(args.device)
                cur_len = len(tokenizer.encode(prompt))
                max_length = args.length
                past_key_values = None
                prev = None
                input_text = input_text.expand(args.samples, input_text.shape[-1])
                result = input_text[:, input_text.shape[-1] - cur_len:]
                while cur_len < max_length:
                    if past_key_values is None:
                        dic = model(input_text, return_dict=True, use_cache=True, config=config, att_tokens_ids=att_tokens_ids)
                        logits, past_key_values = dic.logits, dic.past_key_values
                    else:
                        dic = model(prev, past_key_values=past_key_values, return_dict=True, use_cache=True, config=config, att_tokens_ids=att_tokens_ids)
                        logits, past_key_values = dic.logits, dic.past_key_values
                    probs = torch.softmax(logits[:, -1, :], dim=-1)
                    top_probs, top_indices = torch.topk(probs, args.topk, dim=-1)
                    tmp_prev = torch.multinomial(top_probs, num_samples=1)
                    cur_len += 1
                    prev = torch.gather(top_indices, dim=-1, index=tmp_prev)
                    result = torch.cat([result, prev], dim=-1)
            clean_res = []
            for i in range(args.samples):
                clean_res.append(tokenizer.decode(result[i]))
            for text in clean_res:
                data = {}
                data['text'] = text
                for key in list(comb.keys()):
                    data[key] = comb[key]
                if comb in seen_combs:
                    json.dump(data, file_seen)
                    file_seen.write('\n')
                elif comb in unseen_combs:
                    json.dump(data, file_unseen)
                    file_unseen.write('\n')
                else:
                    raise Exception("Wrong type")
    file_seen.close()
    file_unseen.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--output_data_dir", default=None)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--learning_rate", default=7.5e-5, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=3, type=int)
    parser.add_argument("--warmup_rate", default=0.1, type=float)
    parser.add_argument("--logging_steps", default=50, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_pseu", default=7, type=int)
    parser.add_argument("--dcg_att_len", default=6, type=int)
    parser.add_argument("--dcg_task_len", default=44, type=int)
    parser.add_argument("--alpha", default=0.1, type=float)
    parser.add_argument("--lambda_s", default=None, type=float)
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--unseen_combs_path", default=None)
    parser.add_argument("--dataset_path", default=None, type=str)
    parser.add_argument("--meta_mctg", action='store_true')
    parser.add_argument("--sample_train", action='store_true', help='use sample_train you will sample combs and then sample data with the combs to train your model')
    parser.add_argument("--num_sample_combs", default=None, type=int, help='if use sample_train, then num_sample_combs can be applied')

    parser.add_argument("--length", default=50, type=int)
    parser.add_argument("--samples", default=10, type=int)
    parser.add_argument("--prompt", default=['Once upon a time', 'The book', 'The chicken', 'The city', 'The country', 'The horse', 'The lake', 'The last time', 'The movie', 'The painting', 'The pizza', 'The potato', 'The president of the country', 'The road', 'The year is 1910.', 'In summary', 'This essay discusses', 'Views on', 'The connection', 'Foundational to this is', 'To review,', 'In brief,', 'An illustration of', 'Furthermore,', 'The central theme', 'To conclude,', 'The key aspect', 'Prior to this', 'Emphasised are', 'To summarise', 'The relationship', 'More importantly,', 'It has been shown', 'The issue focused on', 'In this essay'], type=str)
    parser.add_argument("--topk", default=200, type=int)
    parser.add_argument("--test_seed", default=1, type=int)
    parser.add_argument("--device_num", default=0, type=int)
    parser.add_argument("--mode", default=None, type=str, choices=['Hold-Out', 'ACD', 'Few-Shot', 'Original'])
    parser.add_argument("--idx", default=None, type=int)

    args = parser.parse_args()
    assert args.dataset is not None

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.output_data_dir):
        os.makedirs(args.output_data_dir)

    args.device = torch.device("cuda:{}".format(args.device_num))

    unseen_combs_dict = {}
    f = open(args.unseen_combs_path, 'r')
    for item in f.readlines():
        dic = json.loads(item)
        unseen_combs = dic['unseen_combs']
        idx = dic['idx']
        mode = dic['mode']
        if mode not in list(unseen_combs_dict.keys()):
            unseen_combs_dict[mode] = list()
            unseen_combs_dict[mode].append((unseen_combs, mode, idx))
        else:
            unseen_combs_dict[mode].append((unseen_combs, mode, idx))
    f.close()

    train_dataset, mode_name , all_combs, unseen_combs = get_train_dataset(dataset_path=args.dataset_path, unseen_combs_path=args.unseen_combs_path, mode=args.mode, idx=args.idx)
    seen_combs = [i for i in all_combs if i not in unseen_combs]
    args.all_combs = all_combs
    args.seen_combs = seen_combs
    args.unseen_combs = unseen_combs
    args.mode_name = mode_name
    args.train_dataset = train_dataset

    train(args)

if __name__ == "__main__":
    main()
