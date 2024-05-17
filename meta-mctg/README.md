# CompMCTG Benchmark \& Meta-MCTG
## 1. Training Common DCG
```python
python dcg_meta.py --model_name_or_path 'gpt2-medium' \
 --output_dir '../ckpt' \
 --output_data_dir '../test_data' \
 --num_train_epochs 3 \
 --batch_size 8 \
 --dataset 'Amazon' \
 --unseen_combs_path '../data/Amazon/unseen.jsonl' \
 --dataset_path '../data/Amazon/gen.jsonl' \
 --device_num 0 \
 --mode 'Hold-Out' \
 --idx 1
```

## 2. Training Meta DCG
```
python dcg_meta.py --model_name_or_path 'gpt2-medium' \
 --output_dir '../ckpt' \
 --output_data_dir '../test_data' \
 --num_train_epochs 3 \
 --batch_size 8 \
 --dataset 'Amazon' \
 --unseen_combs_path '../data/Amazon/unseen.jsonl' \
 --dataset_path '../data/Amazon/gen.jsonl' \
 --device_num 0 \
 --mode 'Hold-Out' \
 --idx 1 \
 --meta_mctg \
 --lambda_s 0.01
```

## 3. Training Meta DCG (sample-training)
When the number of seen attribute combinations is smaller than mini-batch, using basic Meta DCG is not efficient
