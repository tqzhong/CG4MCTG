# CompMCTG Benchmark \& Meta-MCTG
## 1. Training Common DCG
```python
python dcg_meta.py --model_name_or_path 'gpt2-medium' \
 --output_dir '../ckpt' \
 --output_data_dir '../test_data' \
 --num_train_epochs 3 \
 --dataset 'Amazon' \
 --unseen_combs_path '../data/Amazon/unseen.jsonl' \
 --dataset_path '../data/Amazon/gen.jsonl' \
 --device_num 0 \
 --mode 'Hold-Out' \
 --idx 1
```
