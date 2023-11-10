from cProfile import label
from re import T
import sys
sys.path.append('..')
import torch
from transformers import GPT2Tokenizer, GPT2Config
from modeling_gpt2 import GPT2LMHeadModel
from transformers import get_linear_schedule_with_warmup, AdamW
import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from dataset_benchmark.load_dataset import GenDataset
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
import shutil
import pdb


def tokenize(dataset_path:list, tokenizer) -> list:
    '''
    tokenize the data
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

def padding_fuse_fn(data_list):
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
    
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def const_combs(dataset_path:str) -> None:
    train_data = GenDataset(file_path=dataset_path)
    all_combs = train_data.combs
    all_unseen_combs = open("./Fyelp-v4_unseen_combinations.jsonl", 'w')

    # heldout = 1
    for i, comb in enumerate(all_combs):
        new_dic = {}
        new_dic['unseen_combs'] = [comb]
        new_dic['mode'] = 'heldout=1'
        new_dic['idx'] = '-' + str(i) 
        json.dump(new_dic, all_unseen_combs)
        all_unseen_combs.write('\n')
    # heldout = 2
    combs_heldout2 = list()
    for i, comb_1 in enumerate(all_combs):
        for j, comb_2 in enumerate(all_combs):
            if comb_2 == comb_1:
                continue
            if [comb_2, comb_1] in combs_heldout2:
                continue
            combs_heldout2.append([comb_1, comb_2])
            new_dic = {}
            new_dic['unseen_combs'] = [comb_1, comb_2]
            new_dic['mode'] = 'heldout=2'
            new_dic['idx'] = '-' + str(i) + '-' + str(j)
            json.dump(new_dic, all_unseen_combs)
            all_unseen_combs.write('\n')
    # MCD
    train_data.create_combs_mcd_splits()
    mcd_max_splits = train_data.max_splits
    mcd_rand_splits = train_data.rand_splits
    mcd_min_splits = train_data.min_splits
    for i, (seen_combs, unseen_combs) in enumerate(mcd_max_splits):
        new_dic = {}
        new_dic['unseen_combs'] = unseen_combs
        new_dic['mode'] = 'mcd-max'
        new_dic['idx'] = '-' + str(i)
        json.dump(new_dic, all_unseen_combs)
        all_unseen_combs.write('\n')
    for i, (seen_combs, unseen_combs) in enumerate(mcd_rand_splits):
        new_dic = {}
        new_dic['unseen_combs'] = unseen_combs
        new_dic['mode'] = 'mcd-rand'
        new_dic['idx'] = '-' + str(i)
        json.dump(new_dic, all_unseen_combs)
        all_unseen_combs.write('\n')
    for i, (seen_combs, unseen_combs) in enumerate(mcd_min_splits):
        new_dic = {}
        new_dic['unseen_combs'] = unseen_combs
        new_dic['mode'] = 'mcd-min'
        new_dic['idx'] = '-' + str(i)
        json.dump(new_dic, all_unseen_combs)
        all_unseen_combs.write('\n')
    # few shot
    train_data.create_train_fewshot_split(shot_num=1)
    fewshot_max_splits = train_data.fewshot_max_splits
    fewshot_rand_splits = train_data.fewshot_rand_splits
    fewshot_min_splits = train_data.fewshot_min_splits
    for i, (seen_combs, unseen_combs) in enumerate(fewshot_max_splits):
        new_dic = {}
        new_dic['unseen_combs'] = unseen_combs
        new_dic['mode'] = 'fewshot=1-max'
        new_dic['idx'] = '-' + str(i)
        json.dump(new_dic, all_unseen_combs)
        all_unseen_combs.write('\n')
    for i, (seen_combs, unseen_combs) in enumerate(fewshot_rand_splits):
        new_dic = {}
        new_dic['unseen_combs'] = unseen_combs
        new_dic['mode'] = 'fewshot=1-rand'
        new_dic['idx'] = '-' + str(i)
        json.dump(new_dic, all_unseen_combs)
        all_unseen_combs.write('\n')
    for i, (seen_combs, unseen_combs) in enumerate(fewshot_min_splits):
        new_dic = {}
        new_dic['unseen_combs'] = unseen_combs
        new_dic['mode'] = 'fewshot=1-min'
        new_dic['idx'] = '-' + str(i)
        json.dump(new_dic, all_unseen_combs)
        all_unseen_combs.write('\n')
    
    all_unseen_combs.close()


def get_support_combs(current_combs: list=None, seen_combs: list=None, label_keys: list=None) -> list:
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

def train(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(args.seed)

    config = GPT2Config.from_pretrained(args.model_name_or_path)
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    except:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    args.tokenizer = tokenizer

    train_dataset = GenDataset(file_path=args.dataset_path)
    all_combs = train_dataset.combs
    train_dataset.create_train_by_combs(hold_combs=args.unseen_combs)
    seen_combs = train_dataset.seen_combs
    dataset_path = train_dataset.train
    # pdb.set_trace()
    
    tokenized_data = tokenize(dataset_path=dataset_path, tokenizer=args.tokenizer)
    train_dataset = tokendataset(tokenized_data)
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=False, collate_fn=padding_fuse_fn, sampler=train_sampler)

    # pdb.set_trace()
    # store data by labels (combination)
    label_to_tokenized_data = defaultdict(deque)
    for item in tokenized_data:
        label_to_tokenized_data[json.dumps(item['comb'])].append(item)

    seen_att_tokens_ids = list()
    for item in seen_combs:
        att_tokens_ids = list()
        for key in list(item.keys()):
            assert len(tokenizer.encode(' ' + item[key])) == 1
            att_tokens_ids.append(tokenizer.encode(' ' + item[key])[0])
        seen_att_tokens_ids.append(att_tokens_ids)
    
    label_keys = list(all_combs[0].keys())
    # pdb.set_trace()
    
    # if args.num_pseu >= len(seen_att_tokens_ids) / 2:
    #     args.num_pseu = int(len(seen_att_tokens_ids) / 2)
    if args.num_pseu >= len(seen_att_tokens_ids):
        args.num_pseu = len(seen_att_tokens_ids)
    
    config.is_dcg = True
    config.dcg_att_num = len(label_keys)
    config.dcg_att_len = args.dcg_att_len
    config.dcg_task_len = args.dcg_task_len
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path, config=config)

    if args.model_name_or_path != "/home/zhongtq/pretrained_lms/GPT2-m/":
        addition_epoch = int(args.model_name_or_path.split('-')[-1])
    else:
        addition_epoch = 0

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
    model.train()
    model.zero_grad()
    loss_fct = CrossEntropyLoss()
    args.loss_fct = loss_fct
    logger.info('start_training')
    
    # pdb.set_trace()
    
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

            # pdb.set_trace()
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

                current_combs = [current_combs[i] for i in range(len(current_combs)) if current_combs[i] not in current_combs[:i]]
                support_combs = get_support_combs(current_combs=current_combs, seen_combs=seen_combs, label_keys=label_keys)
                if len(support_combs) == 0:
                    pass
                else:
                    # sample support data (batch)
                    support_data = list()
                    support_num = args.batch_size * args.gradient_accumulation_steps
                    if support_num <= len(support_combs):
                        sample_combs = random.sample(support_combs, support_num)
                        for comb in sample_combs:
                            data_list = label_to_tokenized_data[json.dumps(comb)]
                            sample_data = random.choice(data_list)
                            support_data.append(sample_data)
                    else:
                        support_num_per_comb = int(support_num / len(support_combs))
                        for comb in support_combs:
                            data_list = label_to_tokenized_data[json.dumps(comb)]
                            sample_data = random.sample(data_list, support_num_per_comb)
                            support_data.extend(sample_data)
                        residual_support_num = support_num - support_num_per_comb * len(support_combs)
                        if residual_support_num != 0:
                            data_lists = [label_to_tokenized_data[json.dumps(comb)] for comb in support_combs]
                            combined_data_list = [item for sublist in data_lists for item in sublist]
                            for sampled_data in support_data:
                                combined_data_list.remove(sampled_data)
                            support_data.extend(random.sample(combined_data_list, residual_support_num))
                    assert len(support_data) == support_num
                
                    support_sampler = torch.utils.data.RandomSampler(support_data)
                    support_dataloader = DataLoader(tokendataset(support_data), batch_size=args.batch_size, drop_last=False, collate_fn=padding_fuse_fn, sampler=support_sampler)
                    
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

                        loss_support = args.lambda_s * loss_support_lm
                        loss_support.backward()

                        tr_sloss += loss_support.item()

                    # for param in zip(model.named_parameters(), model.parameters()):
                    #     print(param[0][0],':',param[1].grad)
                    for param, backup_param in zip(model.parameters(), backup_model.parameters()):
                        if backup_param.grad is not None:
                            if param.grad is not None:
                                param.grad += backup_param.grad
                            else:
                                param.grad = backup_param.grad.clone()
                    # for param in zip(model.named_parameters(), model.parameters()):
                    #     print(param[0][0],':',param[1].grad)
                    
                current_combs = list()


                # pdb.set_trace()

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                logs = {}
                loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                loss_lm_scalar = (tr_loss_lm - logging_loss_lm) / args.logging_steps
                loss_dis_scalar = (tr_loss_dis - logging_loss_dis) / args.logging_steps
                loss_s_scalar = (tr_sloss - logging_sloss) / args.logging_steps
                logs['epoch'] = current_epoch
                logs['step'] = global_step
                logs['loss'] = loss_scalar
                logs['lm_loss'] = loss_lm_scalar
                logs['dis_loss'] = loss_dis_scalar
                logs['sloss'] = loss_s_scalar
                logs['lr'] = optimizer.param_groups[0]['lr']
                logging_loss = tr_loss
                logging_loss_lm = tr_loss_lm
                logging_loss_dis = tr_loss_dis
                logging_sloss = tr_sloss
                print(logs)
            
        if current_epoch <= args.num_train_epochs:
            output_dir = os.path.join(args.output_dir, 'dcg_meta-{}-{}-bs-{}-epoch-{}'.format(args.dataset_name, args.mode_name, args.batch_size * args.gradient_accumulation_steps, current_epoch + addition_epoch))
            model_to_save = (model.module if hasattr(model, 'module') else model)
            model_to_save.save_pretrained(output_dir)
            config.save_pretrained(output_dir)

        args.finetuned_model = output_dir
        args.epoch_name = 'dcg_meta-{}-{}-bs-{}-epoch-{}'.format(args.dataset_name, args.mode_name, args.batch_size * args.gradient_accumulation_steps, current_epoch + addition_epoch)

        # generate a version per epoch
        test(args)

    
    logger.info(' global_step = %s, average loss = %s', global_step, tr_loss / global_step)

def test(args):
    set_seed(args.test_seed)
    
    model = GPT2LMHeadModel.from_pretrained(args.finetuned_model).to(args.device)
    config = GPT2Config.from_pretrained(args.finetuned_model)
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    except:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    args.tokenizer = tokenizer
    model.eval()

    '''
    construct all the possible combinations include seen_type (in the training dataset) and the unseen_type
    '''

    train_data = GenDataset(file_path=args.dataset_path)
    all_combs = train_data.combs
    seen_combs = [i for i in all_combs if i not in args.unseen_combs]
    unseen_combs = args.unseen_combs

    if 'epoch_name' in args:
        if args.epoch_name != None:
            file_seen = open(os.path.join(args.output_data_dir, "{}_seen.jsonl".format(args.epoch_name)), 'w')
            file_unseen = open(os.path.join(args.output_data_dir, "{}_unseen.jsonl".format(args.epoch_name)), 'w')
        else:
            file_seen = open(os.path.join(args.output_data_dir, "dcg_meta_{}_{}_seen.jsonl".format(args.dataset_name, args.mode_name)), 'w')
            file_unseen = open(os.path.join(args.output_data_dir, "dcg_meta_{}_{}_unseen.jsonl".format(args.dataset_name, args.mode_name)), 'w')
    else:
        file_seen = open(os.path.join(args.output_data_dir, "dcg_meta_{}_{}_seen.jsonl".format(args.dataset_name, args.mode_name)), 'w')
        file_unseen = open(os.path.join(args.output_data_dir, "dcg_meta_{}_{}_unseen.jsonl".format(args.dataset_name, args.mode_name)), 'w')

    for prompt in tqdm(args.prompt):
        for comb in all_combs:
            att_tokens_ids = list()
            for key in list(comb.keys()):
                # the encoding length of attribute token = 1
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
                        # pdb.set_trace()
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
    parser.add_argument("--model_name_or_path", default="/home/zhongtq/pretrained_lms/GPT2-m/", type=str)
    parser.add_argument("--output_dir", default="../ckpt/dcg_meta/Fyelp-v3/", type=str)
    parser.add_argument("--output_data_dir", default="../test_data/dcg_meta/Fyelp-v3/")
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
    parser.add_argument("--lambda_s", default=0.7, type=float)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--dataset_name", default=None, type=str)
    parser.add_argument("--unseen_combs_path", default="./unseen_combinations/Fyelp-v3_unseen_combinations.jsonl")
    parser.add_argument("--dataset_path", default="/home/zhongtq/cg4ctg/dataset_benchmark/dataset/Fyelp-v3/gen.jsonl", type=str)
    parser.add_argument("--finetuned_model", default=None, type=str)
    parser.add_argument("--epoch_name", default=None, type=str)
    parser.add_argument("--mode_name", default=None, type=str)
    parser.add_argument("--unseen_combs", default=None)

    parser.add_argument("--length", default=50, type=int)
    parser.add_argument("--samples", default=10, type=int)
    parser.add_argument("--prompt", default=['Once upon a time', 'The book', 'The chicken', 'The city', 'The country', 'The horse', 'The lake', 'The last time', 'The movie', 'The painting', 'The pizza', 'The potato', 'The president of the country', 'The road', 'The year is 1910.', 'In summary', 'This essay discusses', 'Views on', 'The connection', 'Foundational to this is', 'To review,', 'In brief,', 'An illustration of', 'Furthermore,', 'The central theme', 'To conclude,', 'The key aspect', 'Prior to this', 'Emphasised are', 'To summarise', 'The relationship', 'More importantly,', 'It has been shown', 'The issue focused on', 'In this essay'], type=str)
    parser.add_argument("--topk", default=200, type=int)
    parser.add_argument("--test_seed", default=1, type=int)
    parser.add_argument("--device_num", default=6, type=int)
    parser.add_argument("--mode", default=None, type=str, choices=['heldout=1', 'heldout=2', 'mcd-max', 'mcd-rand', 'mcd-min', 'fewshot=1-max', 'fewshot=1-rand', 'fewshot=1-min', 'iid'])
    parser.add_argument("--idx", default=None, type=int)

    args = parser.parse_args()

    assert args.dataset is not None
    args.output_dir = "../ckpt/dcg_meta/{}/".format(args.dataset)
    args.output_data_dir = "../test_data/dcg_meta/{}/".format(args.dataset)
    args.unseen_combs_path = "./unseen_combinations/{}_unseen_combinations.jsonl".format(args.dataset)
    args.dataset_path = "/home/zhongtq/cg4ctg/dataset_benchmark/dataset/{}/gen.jsonl".format(args.dataset)

    args.device = torch.device("cuda:{}".format(args.device_num))
    assert args.unseen_combs_path.split('/')[-1].split('_')[0] == args.dataset_path.split('/')[-2]
    args.dataset_name = args.dataset_path.split('/')[-2]

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
    
    if args.train:
        assert args.mode is not None
        assert args.idx is not None
        if args.mode != 'iid':
            assert args.mode in list(unseen_combs_dict.keys())
            assert args.idx < len(unseen_combs_dict[args.mode])
            args.unseen_combs = unseen_combs_dict[args.mode][args.idx][0]
            args.mode_name = unseen_combs_dict[args.mode][args.idx][1] + unseen_combs_dict[args.mode][args.idx][2]
        else:
            assert args.idx == 0
            args.unseen_combs = list()
            args.mode_name = args.mode
    else:
        assert args.finetuned_model is not None
        assert args.epoch_name is not None
        if args.unseen_combs is None:
            mode = args.epoch_name.split('-')[3]
            if mode == 'heldout=1':
                idx = int(args.epoch_name.split('-')[4])
            elif mode == 'mcd' or mode == 'fewshot=1':
                mode = mode + '-' + args.epoch_name.split('-')[4]
                idx = int(args.epoch_name.split('-')[5])
            elif mode == 'iid':
                idx = 0
            else:
                raise Exception("mode error")

        if mode != 'iid':
            args.unseen_combs = unseen_combs_dict[mode][idx][0]
            args.mode_name = unseen_combs_dict[mode][idx][1] + unseen_combs_dict[mode][idx][2]
        else:
            args.unseen_combs = list()
            args.mode_name = mode
        # print(mode, idx)
        # print(args.unseen_combs)


    if args.train:
        train(args)
    else:
        test(args)
    # const_combs(dataset_path=args.dataset_path)

if __name__ == "__main__":
    main()


