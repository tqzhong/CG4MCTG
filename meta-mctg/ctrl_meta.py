import torch
import sys
sys.path.append('..')
from data.utils import get_train_dataset
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
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

def get_support_combs_without_constraint(current_combs: list=None, seen_combs: list=None, label_keys: list=None) -> list:
    available_combs = seen_combs

    support_combs = [item for item in available_combs if item not in current_combs]
    support_combs = [item for item in support_combs if item in seen_combs]

    return support_combs

logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(args.seed)

    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
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

    tokenized_data = tokenize(dataset_path=dataset_path, tokenizer=args.tokenizer)
    train_dataset = tokendataset(tokenized_data)
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=False, collate_fn=padding_fuse_fn, sampler=train_sampler)

    label_to_tokenized_data = defaultdict(deque)
    for item in tokenized_data:
        label_to_tokenized_data[json.dumps(item['comb'])].append(item)

    label_keys = list(all_combs[0].keys())

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
    tr_sloss, logging_sloss = 0.0, 0.0
    model.train()
    model.zero_grad()
    loss_fct = CrossEntropyLoss(ignore_index=50256)
    args.loss_fct = loss_fct
    logger.info('start_training')

    mini_batch = args.batch_size * args.gradient_accumulation_steps
    if len(seen_combs) < mini_batch:
        args.sample_train = True
    
    if not args.sample_train:

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
                
                multi_ctrl_codes = None
                for key in label_keys:
                    if multi_ctrl_codes is None:
                        multi_ctrl_codes = label_ids[key]
                    else:
                        multi_ctrl_codes = torch.cat([multi_ctrl_codes, label_ids[key]], dim=-1)
                multi_ctrl_codes = multi_ctrl_codes.to(args.device)

                eos_token_ids = torch.tensor(tokenizer.encode(tokenizer.eos_token))
                eos_token_ids = eos_token_ids.expand(args.batch_size, eos_token_ids.shape[0]).to(args.device)

                input_ids = torch.tensor(input_ids).to(args.device)
                input_ids = torch.cat([multi_ctrl_codes, input_ids], dim=-1)
                input_ids = torch.cat([eos_token_ids, input_ids], dim=-1)
                eos_token_mask = torch.tensor([1]).expand(args.batch_size, 1).to(args.device)
                ctrl_codes_mask = torch.tensor([1]).expand(args.batch_size, multi_ctrl_codes.shape[-1]).to(args.device)
                attention_mask = torch.tensor(attention_mask).to(args.device)
                attention_mask = torch.cat([ctrl_codes_mask, attention_mask], dim=-1)
                attention_mask = torch.cat([eos_token_mask, attention_mask], dim=-1)
                
                dic = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, use_cache=True)
                logits = dic.logits
                shift_logits = logits[:, multi_ctrl_codes.shape[-1]:-1, :].contiguous()
                labels = input_ids[:, 1 + multi_ctrl_codes.shape[-1]:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
                loss.backward()
                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    current_combs = [current_combs[i] for i in range(len(current_combs)) if current_combs[i] not in current_combs[:i]]
                    if args.support_constraint:
                        support_combs = get_support_combs(current_combs=current_combs, seen_combs=seen_combs, label_keys=label_keys)
                    else:
                        support_combs = get_support_combs_without_constraint(current_combs=current_combs, seen_combs=seen_combs, label_keys=label_keys)
                    if len(support_combs) == 0:
                        pass
                    else:
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
                            
                            support_multi_ctrl_codes = None
                            for key in label_keys:
                                if support_multi_ctrl_codes is None:
                                    support_multi_ctrl_codes = support_label_ids[key]
                                else:
                                    support_multi_ctrl_codes = torch.cat([support_multi_ctrl_codes, support_label_ids[key]], dim=-1)
                            support_multi_ctrl_codes = support_multi_ctrl_codes.to(args.device)

                            seos_token_ids = torch.tensor(tokenizer.encode(tokenizer.eos_token))
                            seos_token_ids = seos_token_ids.expand(args.batch_size, seos_token_ids.shape[0]).to(args.device)

                            support_input_ids = torch.tensor(support_input_ids).to(args.device)
                            support_input_ids = torch.cat([support_multi_ctrl_codes, support_input_ids], dim=-1)
                            support_input_ids = torch.cat([seos_token_ids, support_input_ids], dim=-1)
                            seos_token_mask = torch.tensor([1]).expand(args.batch_size, 1).to(args.device)
                            sctrl_codes_mask = torch.tensor([1]).expand(args.batch_size, support_multi_ctrl_codes.shape[-1]).to(args.device)
                            support_attention_mask = torch.tensor(support_attention_mask).to(args.device)
                            support_attention_mask = torch.cat([sctrl_codes_mask, support_attention_mask], dim=-1)
                            support_attention_mask = torch.cat([seos_token_mask, support_attention_mask], dim=-1)

                            dic = model(input_ids=support_input_ids, attention_mask=support_attention_mask, return_dict=True, use_cache=True)
                            slogits = dic.logits
                            sshift_logits = slogits[:, support_multi_ctrl_codes.shape[-1]:-1, :].contiguous()
                            slabels = support_input_ids[:, 1+support_multi_ctrl_codes.shape[-1]:].contiguous()
                            sloss = loss_fct(sshift_logits.view(-1, sshift_logits.size(-1)), slabels.view(-1))
                            
                            sloss = args.lambda_s * sloss
                            sloss.backward()

                            tr_sloss += sloss.item()
                        
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
                    loss_s_scalar = (tr_sloss - logging_sloss) / args.logging_steps
                    logs['epoch'] = current_epoch
                    logs['step'] = global_step
                    logs['loss'] = loss_scalar
                    logs['sloss'] = loss_s_scalar
                    logging_loss = tr_loss
                    logging_sloss = tr_sloss
                    print(logs)
            
            if current_epoch <= args.num_train_epochs:
                output_dir = os.path.join(args.output_dir, 'ctrl_meta-{}-{}-lambda={}-bs-{}-epoch-{}'.format(args.dataset, args.mode_name, args.lambda_s, args.batch_size * args.gradient_accumulation_steps, current_epoch))
                model_to_save = (model.module if hasattr(model, 'module') else model)
                model_to_save.save_pretrained(output_dir)
            
                args.finetuned_model = output_dir
                args.epoch_name = 'ctrl_meta-{}-{}-lambda={}-bs-{}-epoch-{}'.format(args.dataset, args.mode_name, args.lambda_s, args.batch_size * args.gradient_accumulation_steps, current_epoch)

                # generate a version per epoch
                test(args)
    
    else:
        print("Applying Sample Training")
        assert len(seen_combs) >= 2
        if args.num_sample_combs is None:
            num_sample_combs = int(len(seen_combs) / 2)
        else:
            num_sample_combs = args.num_sample_combs
        
        assert mini_batch % num_sample_combs == 0
        new_gradient_accumulation_steps = int(mini_batch / num_sample_combs)
        new_batch_size = num_sample_combs

        current_epoch = 0
        for epoch in trange(int(args.num_train_epochs), desc='Epoch'):
            current_epoch += 1
            backup_label_to_tokenized_data = copy.deepcopy(label_to_tokenized_data)
            num_combs_of_sample_combs = math.comb(len(seen_combs), num_sample_combs)
            samples_per_comb_of_sample_combs = int(len(tokenized_data) / num_combs_of_sample_combs / num_sample_combs)
            combs_of_comb_list = list()
            all_combs_of_comb = list(itertools.combinations(seen_combs, num_sample_combs))
            for item in all_combs_of_comb:
                for i in range(samples_per_comb_of_sample_combs):
                    combs_of_comb_list.append(item)
            residual_num_per_comb = int((len(tokenized_data) - samples_per_comb_of_sample_combs * num_combs_of_sample_combs * num_sample_combs) / len(seen_combs))

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

                    multi_ctrl_codes = None
                    for key in label_keys:
                        if multi_ctrl_codes is None:
                            multi_ctrl_codes = label_ids[key]
                        else:
                            multi_ctrl_codes = torch.cat([multi_ctrl_codes, label_ids[key]], dim=-1)
                    multi_ctrl_codes = multi_ctrl_codes.to(args.device)

                    eos_token_ids = torch.tensor(tokenizer.encode(tokenizer.eos_token))
                    eos_token_ids = eos_token_ids.expand(new_batch_size, eos_token_ids.shape[0]).to(args.device)
                    # pdb.set_trace()
                    input_ids = torch.tensor(input_ids).to(args.device)
                    input_ids = torch.cat([multi_ctrl_codes, input_ids], dim=-1)
                    input_ids = torch.cat([eos_token_ids, input_ids], dim=-1)
                    eos_token_mask = torch.tensor([1]).expand(new_batch_size, 1).to(args.device)
                    ctrl_codes_mask = torch.tensor([1]).expand(new_batch_size, multi_ctrl_codes.shape[-1]).to(args.device)
                    attention_mask = torch.tensor(attention_mask).to(args.device)
                    attention_mask = torch.cat([ctrl_codes_mask, attention_mask], dim=-1)
                    attention_mask = torch.cat([eos_token_mask, attention_mask], dim=-1)

                    dic = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, use_cache=True)
                    logits = dic.logits
                    shift_logits = logits[:, multi_ctrl_codes.shape[-1]:-1, :].contiguous()
                    labels = input_ids[:, 1 + multi_ctrl_codes.shape[-1]:].contiguous()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
                    loss.backward()
                    tr_loss += loss.item()

                    if i == new_gradient_accumulation_steps - 1:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                        backup_model = copy.deepcopy(model)
                        backup_model.to(args.device)
                        with torch.no_grad():
                            for param, backup_param in zip(model.parameters(), backup_model.parameters()):
                                if param.grad is not None:
                                    backup_param -= optimizer.param_groups[0]['lr'] * param.grad

                        current_combs = [current_combs[i] for i in range(len(current_combs)) if current_combs[i] not in current_combs[:i]]
                        if args.support_constraint:
                            support_combs = get_support_combs(current_combs=current_combs, seen_combs=seen_combs, label_keys=label_keys)
                        else:
                            support_combs = get_support_combs_without_constraint(current_combs=current_combs, seen_combs=seen_combs, label_keys=label_keys)
                        if len(support_combs) == 0:
                            pass
                        else:
                            support_data = list()
                            support_num = mini_batch
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

                            for support_batch in support_dataloader:
                                support_input_ids, support_attention_mask = support_batch['input_ids'], support_batch['attention_mask']
                                support_label_ids = dict()
                                for key in label_keys:
                                    support_label_ids[key] = torch.tensor(support_batch[key])
                                support_multi_ctrl_codes = None
                                for key in label_keys:
                                    if support_multi_ctrl_codes is None:
                                        support_multi_ctrl_codes = support_label_ids[key]
                                    else:
                                        support_multi_ctrl_codes = torch.cat([support_multi_ctrl_codes, support_label_ids[key]], dim=-1)
                                support_multi_ctrl_codes = support_multi_ctrl_codes.to(args.device)

                                seos_token_ids = torch.tensor(tokenizer.encode(tokenizer.eos_token))
                                seos_token_ids = seos_token_ids.expand(args.batch_size, seos_token_ids.shape[0]).to(args.device)

                                support_input_ids = torch.tensor(support_input_ids).to(args.device)
                                support_input_ids = torch.cat([support_multi_ctrl_codes, support_input_ids], dim=-1)
                                support_input_ids = torch.cat([seos_token_ids, support_input_ids], dim=-1)
                                seos_token_mask = torch.tensor([1]).expand(args.batch_size, 1).to(args.device)
                                sctrl_codes_mask = torch.tensor([1]).expand(args.batch_size, support_multi_ctrl_codes.shape[-1]).to(args.device)
                                support_attention_mask = torch.tensor(support_attention_mask).to(args.device)
                                support_attention_mask = torch.cat([sctrl_codes_mask, support_attention_mask], dim=-1)
                                support_attention_mask = torch.cat([seos_token_mask, support_attention_mask], dim=-1)

                                dic = model(input_ids=support_input_ids, attention_mask=support_attention_mask, return_dict=True, use_cache=True)
                                slogits = dic.logits
                                sshift_logits = slogits[:, support_multi_ctrl_codes.shape[-1]:-1, :].contiguous()
                                slabels = support_input_ids[:, 1+support_multi_ctrl_codes.shape[-1]:].contiguous()
                                sloss = loss_fct(sshift_logits.view(-1, sshift_logits.size(-1)), slabels.view(-1))
                                
                                sloss = args.lambda_s * sloss
                                sloss.backward()

                                tr_sloss += sloss.item()

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
                        loss_s_scalar = (tr_sloss - logging_sloss) / args.logging_steps
                        logs['epoch'] = current_epoch
                        logs['step'] = global_step
                        logs['loss'] = loss_scalar
                        logs['sloss'] = loss_s_scalar
                        logging_loss = tr_loss
                        logging_sloss = tr_sloss
                        print(logs)
            
            if current_epoch <= args.num_train_epochs:
                output_dir = os.path.join(args.output_dir, 'ctrl_sample_meta-{}-{}-lambda={}-bs-{}-epoch-{}'.format(args.dataset, args.mode_name, args.lambda_s, args.batch_size * args.gradient_accumulation_steps, current_epoch))
                model_to_save = (model.module if hasattr(model, 'module') else model)
                model_to_save.save_pretrained(output_dir)
            
                args.finetuned_model = output_dir
                args.epoch_name = 'ctrl_sample_meta-{}-{}-lambda={}-bs-{}-epoch-{}'.format(args.dataset, args.mode_name, args.lambda_s, args.batch_size * args.gradient_accumulation_steps, current_epoch)

                # generate a version per epoch
                test(args)



    logger.info(' global_step = %s, average loss = %s', global_step, tr_loss / global_step)

def test(args):
    set_seed(args.test_seed)

    model = GPT2LMHeadModel.from_pretrained(args.finetuned_model).to(args.device)
    model.eval()
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    except:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    args.tokenizer = tokenizer

    train_data = GenDataset(file_path=args.dataset_path)
    all_combs = train_data.combs
    seen_combs = [i for i in all_combs if i not in args.unseen_combs]
    unseen_combs = args.unseen_combs

    if 'epoch_name' in args:
        file_seen = open(os.path.join(args.output_data_dir, '{}_seen.jsonl'.format(args.epoch_name)), 'w')
        file_unseen = open(os.path.join(args.output_data_dir, '{}_unseen.jsonl'.format(args.epoch_name)), 'w')
    else:
        file_seen = open(os.path.join(args.output_data_dir, 'ctrl_{}_{}_seen.jsonl'.format(args.dataset, args.mode_name)), 'w')
        file_unseen = open(os.path.join(args.output_data_dir, 'ctrl_{}_{}_unseen.jsonl'.format(args.dataset, args.mode_name)), 'w')

    for prompt in tqdm(args.prompt):
        for comb in all_combs:
            multi_ctrl_codes = list()
            for key in list(comb.keys()):
                # pdb.set_trace()
                assert len(tokenizer.encode(' ' + comb[key])) == 1
                multi_ctrl_codes.append(tokenizer.encode(' ' + comb[key])[0])

            with torch.no_grad():
                input_text = torch.tensor(tokenizer.encode(tokenizer.eos_token) + multi_ctrl_codes + tokenizer.encode(prompt)).long().unsqueeze(0).to(args.device)
                cur_len = len(tokenizer.encode(prompt))
                max_length = args.length
                past_key_values = None
                prev = None
                input_text = input_text.expand(args.samples, input_text.shape[-1])
                result = input_text[:, input_text.shape[-1]-cur_len:]
                while cur_len < max_length:
                    if past_key_values is None:
                        dic = model(input_text, return_dict=True, use_cache=True)
                        logits, past_key_values = dic.logits, dic.past_key_values
                    else:
                        dic = model(prev, past_key_values=past_key_values, return_dict=True, use_cache=True)
                        logits, past_key_values = dic.logits, dic.past_key_values
                    probs = torch.softmax(logits[:, -1, :], dim=-1)
                    top_probs, top_indices = torch.topk(probs, args.topk, dim=-1)
                    tmp_prev = torch.multinomial(top_probs, num_samples=1)
                    cur_len += 1
                    prev = torch.gather(top_indices, dim=-1, index=tmp_prev)
                    result = torch.cat((result, prev), dim=-1)
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
    try:
        shutil.rmtree(args.finetuned_model)
    except:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--output_data_dir", default=None, type=str)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument("--warmup_rate", default=0.1, type=float)
    parser.add_argument("--logging_steps", default=50, type=int)
    parser.add_argument("--device_num", default=0, type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lambda_s", default=0.7, type=float)
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--length", default=50, type=int)
    parser.add_argument("--samples", default=10, type=int)
    parser.add_argument("--prompt", default=['Once upon a time', 'The book', 'The chicken', 'The city', 'The country', 'The horse', 'The lake', 'The last time', 'The movie', 'The painting', 'The pizza', 'The potato', 'The president of the country', 'The road', 'The year is 1910.', 'In summary', 'This essay discusses', 'Views on', 'The connection', 'Foundational to this is', 'To review,', 'In brief,', 'An illustration of', 'Furthermore,', 'The central theme', 'To conclude,', 'The key aspect', 'Prior to this', 'Emphasised are', 'To summarise', 'The relationship', 'More importantly,', 'It has been shown', 'The issue focused on', 'In this essay'], type=str)
    parser.add_argument("--topk", default=200, type=int)
    parser.add_argument("--test_seed", default=1, type=int)
    parser.add_argument("--sample_train", action='store_true', help='use sample_train you will sample combs and then sample data with the combs to train your model')
    parser.add_argument("--num_sample_combs", default=None, type=int, help='if use sample_train, then num_sample_combs can be applied')
    parser.add_argument("--support_constraint", action="store_false")
    parser.add_argument("--dataset_path", default=None, type=str)
    parser.add_argument("--unseen_combs_path", default=None, type=str)
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