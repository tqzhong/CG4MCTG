
dataset_path=./test_cases/cases_Fyelp.jsonl
dataset=Fyelp
device_num=0

python ./scripts/eval_compmctg.py --dataset_path $dataset_path --dataset $dataset --device_num $device_num
