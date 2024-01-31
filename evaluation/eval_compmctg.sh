
dataset_path=/home/zhongtq/cg4ctg/test_data/ctrl/ctrl_Fyelp-v3/ctrl_fewshot=1-max_seen.jsonl
dataset=Fyelp
device_num=3

python ./scripts/eval_compmctg.py --dataset_path $dataset_path --dataset $dataset --device_num $device_num