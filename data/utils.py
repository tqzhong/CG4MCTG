import json
import argparse

def check_train(datum:dict, unseen_combs:list) -> bool:
    '''
    return True: datum is supposed to be in train_set;
    otherwise: datum is supposed to be hold out;
    '''
    for i in range(len(unseen_combs)):
        flag = True
        for key in unseen_combs[i].keys():
            if datum[key] != unseen_combs[i][key]:
                flag = False
        if flag == False: # at least one attribute different
            pass
        else: # in this combination, all the attributes match
            return False
    return True

def get_data_by_unseen_combs(dataset_path:str, unseen_combs:list) -> list:
    '''
    dataset_path: contains all data of this dataset
    unseen_combs: contains one or more attribute combinations 
    The functions's purpose is to filter out all data from dataset_path that contains attribute combinations present in unseen_combs.
    '''
    f = open(dataset_path, 'r')
    all_data = list()
    for item in f.readlines():
        all_data.append(json.loads(item))

    data_train = list()
    for datum in all_data:
        if check_train(datum, unseen_combs) == True:
            data_train.append(datum)
    
    return data_train

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="./data/Yelp/gen.jsonl", type=str)
    parser.add_argument("--unseen_combs_path", default="./data/unseen.jsonl", type=str)
    parser.add_argument("--mode", default=None, type=str, choices=['Hold-Out', 'ACD', 'Few-Shot', 'Original'])
    parser.add_argument("--idx", default=None, type=int)
    args = parser.parse_args()

    
    '''
    For Yelp, it contains 8 Hold-Out modes, 10 ACD modes, and 8 Few-Shot modes
    We can create a train_set with specific mode name and its idx
    For example, we choose mode ACD and idx = 1, then we can get train_set as follows
    '''

    args.mode = 'ACD'
    args.idx = 1

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

    assert args.mode in list(unseen_combs_dict.keys())
    assert args.idx < len(unseen_combs_dict[args.mode])
    unseen_combs = unseen_combs_dict[args.mode][args.idx][0]

    train_set = get_data_by_unseen_combs(dataset_path=args.dataset_path, unseen_combs=unseen_combs)

    mode_name = unseen_combs_dict[args.mode][args.idx][1] + unseen_combs_dict[args.mode][args.idx][2]