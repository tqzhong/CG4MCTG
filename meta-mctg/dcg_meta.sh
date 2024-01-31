
model_name_or_path=gpt2-medium
output_dir=../temps
output_data_dir=../temps
num_train_epochs=3
lambda_s=0.01
dataset=Amazon
unseen_combs_path=../data/Amazon/unseen.jsonl
dataset_path=../data/Amazon/gen.jsonl
device_num=4
mode=Hold-Out
idx=1
num_sample_combs=2

# common-training ——base dcg
python dcg_meta.py --model_name_or_path $model_name_or_path \
                   --output_dir $output_dir \
                   --output_data_dir $output_data_dir \
                   --num_train_epochs $num_train_epochs \
                   --dataset $dataset \
                   --unseen_combs_path $unseen_combs_path \
                   --dataset_path $dataset_path \
                   --device_num $device_num \
                   --mode $mode \
                   --idx $idx \

# meta-mctg-training ——meta dcg
python dcg_meta.py --model_name_or_path $model_name_or_path \
                   --output_dir $output_dir \
                   --output_data_dir $output_data_dir \
                   --num_train_epochs $num_train_epochs \
                   --dataset $dataset \
                   --unseen_combs_path $unseen_combs_path \
                   --dataset_path $dataset_path \
                   --device_num $device_num \
                   --mode $mode \
                   --idx $idx \
                   --lambda_s $lambda_s \
                   --meta_mctg

# meta-mctg-sample-training (when the number of seen combinations is smaller than mini-batch, choose use meta-mctg sample training) ——meta dcg
# What is "meta-mctg-sample-training"? 
# It is a method that control the number of combinations in the train batch to a certain value (i.e.,"num_sample_combs") so that we can construct the pseudo-comp batch easier when the number of seen combinations is smaller than mini-batch. For example, in ACD of YELP, the number of seen combinations is 4, which is smaller than mini-batch that equals to 8, if we use common "meta-mctg-training", we will sample a train batch randomly from the dataloader and the number of combinations in the train batch is going to be high probability equal to four (or three whatever). In this case, there will be no other combination to build a pseudo-comp batch.

# note: When the number of seen combinations is smaller than mini-batch, script will enforce the use of meta-mctg sample training. When the number of seen combinations is equal to mini-batch or larger than mini-batch, you can choose using meta-mctg training or meta-mctg sample training. But when you choose using meta-mctg sample training, you need to set the value of "num_sample_combs".

# In our experience, when the number of seen combinations is larger than mini-batch (e.g.,Hold-Out/ACD of Fyelp or Hold-Out of Amazon), "meta-mctg-training" produce good enough results. In this situation, we prefer using "meta-mctg-training". However, it is not ruled out that sometimes "meta-mctg-sample-training" will have better results. 
# When the number of seen combinations is smaller than mini-batch (e.g., Hold-Out/ACD of Yelp or Hold-Out of Mixture), "meta-mctg-training" almost not work because we can hardly construct the "pseudo-comp batch" based on a ramdom sampled train batch. In this situation, we always use "meta-mctg-sample-training" and we set the value of "num_sample_combs" to 2 for most of the time. It is worth noting that the value of "num_sample_combs" connot be set to 1, as this connot construct the "pseudo-comp batch" either. (no compositional generalization when the number of combinations in "train batch" is 1)

python dcg_meta.py --model_name_or_path $model_name_or_path \
                   --output_dir $output_dir \
                   --output_data_dir $output_data_dir \
                   --num_train_epochs $num_train_epochs \
                   --dataset $dataset \
                   --unseen_combs_path $unseen_combs_path \
                   --dataset_path $dataset_path \
                   --device_num $device_num \
                   --mode $mode \
                   --idx $idx \
                   --lambda_s $lambda_s \
                   --meta_mctg \
                   --sample_train \
                   --num_sample_combs $num_sample_combs
