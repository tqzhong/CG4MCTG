from model import RobertaForPreTraining
from transformers import RobertaTokenizer, RobertaConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import torch
import json
import pdb

MAXLEN = 512
BATCH_SIZE = 4
# DEVICE = torch.device("cuda")

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

    key_list = list(data_list[0].keys())
    key_list.remove('text')
    for key in key_list:
        labels[key] = list()

    for item in data_list:
        text_length.append(len(item['text']))
        for key in key_list:
            labels[key].append([item[key]])
    max_text_len = max(text_length)
    for i, item in enumerate(data_list):
        text_pad_len = max_text_len - text_length[i]
        attention_mask = [1] * text_length[i] + [0] * text_pad_len
        text = item["text"] + [0] * text_pad_len

        input_ids.append(text)
        attention_masks.append(attention_mask)
    
    batch = dict()
    batch['input_ids'] = input_ids
    batch['attention_mask'] = attention_masks
    for key in key_list:
        batch[key] = labels[key]

    return batch

def tokenize(dataset_path:list, tokenizer) -> list:
    '''
    tokenize the data
    '''
    tokenized_data = list()
    for dic in dataset_path:
        new_dic = {}
        if 'text' in dic:
            new_dic['text'] = tokenizer.encode(dic['text'], max_length=512, truncation=True)
        elif 'review' in dic:
            new_dic['text'] = tokenizer.encode(dic['review'], max_length=512, truncation=True)
        attribute_keys = list(dic.keys())[:]
        if 'text' in dic:
            attribute_keys.remove('text')
        elif 'review' in dic:
            attribute_keys.remove('review')
        if 'type' in dic:
            attribute_keys.remove('type')
        for key in attribute_keys:
            value = dic[key]
            if value == 'Negative':
                new_dic[key] = 0
            elif value == 'Positive':
                new_dic[key] = 1
            if value == 'Female':
                new_dic[key] = 0
            elif value == 'Male':
                new_dic[key] = 1
            if value == 'Asian':
                new_dic[key] = 0
            elif value == 'American':
                new_dic[key] = 1
            elif value == 'Mexican':
                new_dic[key] = 2
            elif value == 'Bar':
                new_dic[key] = 3
            elif value == 'dessert':
                new_dic[key] = 4
            if value == 'Present':
                new_dic[key] = 0
            elif value == 'Past':
                new_dic[key] = 1
        tokenized_data.append(new_dic)

    return tokenized_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--dataset_path", default=None, type=str)
    parser.add_argument("--batch_size", default=BATCH_SIZE, type=int)
    parser.add_argument("--tar_dim", default=None, type=int)
    parser.add_argument("--test_aspect", default=None, type=str, choices=['sentiment', 'gender', 'cuisine', 'tense'])
    parser.add_argument("--device_num", default=None, type=str)
    args = parser.parse_args()
    # args.device = DEVICE
    args.device = torch.device("cuda:{}".format(args.device_num))

    aspect_list = ['sentiment', 'gender', 'cuisine', 'tense']
    acc_dic = dict()
    for aspect in aspect_list:
        args.test_aspect = aspect

        # assert args.test_aspect is not None
        if args.test_aspect == 'sentiment':
            args.model_name_or_path = './classifiers/Fyelp/sentiment'
            args.tar_dim = 2
        elif args.test_aspect == 'gender':
            args.model_name_or_path = './classifiers/Fyelp/gender'
            args.tar_dim = 2
        elif args.test_aspect == 'cuisine':
            args.model_name_or_path = './classifiers/Fyelp/cuisine'
            args.tar_dim = 5
        elif args.test_aspect == 'tense':
            args.model_name_or_path = './classifiers/Fyelp/tense'
            args.tar_dim = 2

        # tokenized the data in dataset_path, original roberta tokenizer
        config = RobertaConfig.from_pretrained(args.model_name_or_path)
        assert args.tar_dim is not None
        config.tar_dim = args.tar_dim
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)

        f = open(args.dataset_path, 'r')
        all_data = list()
        label_keys = list()
        for item in f.readlines():
            dic = json.loads(item)
            all_data.append(dic)

            if len(label_keys) == 0:
                keys = list(dic.keys())
                if 'text' in dic:
                    keys.remove('text')
                elif 'review' in dic:
                    keys.remove('review')
                if 'type' in dic:
                    keys.remove('type')
                label_keys = keys
        
        assert args.test_aspect in label_keys
        # if 'topic_cged' in label_keys:
        #     label_keys.remove('topic_cged')

        tokenized_data = tokenize(dataset_path=all_data, tokenizer=tokenizer)
        test_dataset = tokendataset(tokenized_data)
        test_sampler = torch.utils.data.RandomSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False, collate_fn=padding_fuse_fn, sampler=test_sampler)

        model = RobertaForPreTraining.from_pretrained(args.model_name_or_path, config=config)
        model.to(args.device)
        model.eval()

        tp_all = 0
        fp_all = 0
        tr_loss = 0.0
        acc_part_all = dict()
        acc_part_all['tp'] = dict()
        acc_part_all['fp'] = dict()
        for i in range(args.tar_dim):
            acc_part_all['tp'][i] = 0
            acc_part_all['fp'][i] = 0
        logs = {}
        
        acc_lst = dict()

        for step, batch in enumerate(test_dataloader):
            input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
            label_ids = dict()

            for key in label_keys:
                label_ids[key] = torch.tensor(batch[key]).to(args.device)

            input_ids = torch.tensor(input_ids).to(args.device)
            attention_mask = torch.tensor(attention_mask).to(args.device)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0

            # pdb.set_trace()
            loss, accuracy, acc_part = model(input_ids=input_ids, attention_mask=attention_mask, label=label_ids[args.test_aspect])
            tp_all += accuracy
            tr_loss += loss.item()

            for i in range(config.tar_dim):
                fp_all += acc_part['fp'][i]
            for i in range(config.tar_dim):
                acc_part_all['tp'][i] += acc_part['tp'][i]
                acc_part_all['fp'][i] += acc_part['fp'][i]
        
        acc = tp_all / (tp_all + fp_all)

        acc_dic['acc_{}'.format(aspect)] = float('{:.4f}'.format(acc))
    
    acc_avg = 0
    for key in aspect_list:
        acc_avg += acc_dic['acc_{}'.format(key)]
    acc_avg = acc_avg / len(aspect_list)
    acc_dic['acc_avg'] = float('{:.4f}'.format(acc_avg))

    for key in list(acc_dic.keys()):
        print({key: acc_dic[key]})
        
        # for i in range(config.tar_dim):
        #     acc_lst[i] = acc_part_all['tp'][i] / (acc_part_all['tp'][i] + acc_part_all['fp'][i])
        # logs['acc'] = float('{:.4f}'.format(acc))
        # for i in range(config.tar_dim):
        #     logs['acc{}'.format(i)] = float('{:.4f}'.format(acc_lst[i]))
        # logs['total_loss'] = tr_loss
        # print(logs)

if __name__ == "__main__":
    main()
