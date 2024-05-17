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

## 2. Training Meta DCG (meta-mctg-training)
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

## 3. Training Meta DCG (meta-mctg-sample-training)
When the number of seen attribute combinations is smaller than mini-batch, using basic Meta DCG is not efficient. Meta-mctg-sample-training is a method that control the number of attribute combinations in the train batch to a certain value (i.e., hyperparameter "num_sample_combs") so that we can construct the pseudo-comp batch easier when the number of seen combinations is smaller than mini-batch. For example, in ACD of YELP, the number of seen attribute combinations is 4, which is smaller than mini-batch that equals to 8, if we use common "meta-mctg-training", we will sample a train batch randomly from the dataloader and the number of attribute combinations in the train batch is going to be high probability equal to four (or three whatever). In this casem there will be no other combination to build a pseudo-comp batch.

Note: When the number of seen attribute combinations is smaller than mini-batch, script will enforce the use of meta-mctg sample training. When the number of seen attribute combinations is equal to mini-batch or larger than mini-batch, you can choose using meta-mctg-training or meta-mctg-sample-training. But when you choose using meta-mctg-sample-training, you need to set the value of "num_sample_combs".

In our experience, when the number of seen attribtue combinations is larger than mini-batch (e.g.,Hold-Out/ACD of Fyelp or Hold-Out of Amazon), "meta-mctg-training" produce good enough results. In this situation, we prefer using "meta-mctg-training". However, it is not ruled out that sometimes "meta-mctg-sample-training" will have better results. 

When the number of seen attribute combinations is smaller than mini-batch (e.g., Hold-Out/ACD of Yelp or Hold-Out of Mixture), "meta-mctg-training" almost not work because we can hardly construct the "pseudo-comp batch" based on a ramdom sampled train batch. In this situation, we always use "meta-mctg-sample-training" and we set the value of "num_sample_combs" to 2 for most of the time. It is worth noting that the value of "num_sample_combs" connot be set to 1, as this connot construct the "pseudo-comp batch" either (no compositional generalization when the number of combinations in "train batch" is 1).

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
 --lambda_s 0.01 \
 --sample_train \
 --num_sample_combs 2
``` 
