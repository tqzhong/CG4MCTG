# CompMCTG Benchmark \& Meta-MCTG
## 1. Training Common MCTG
```shell
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

## 2. Training Meta MCTG (meta-mctg-training)
```shell
python dcg_meta.py --model_name_or_path 'gpt2-medium' \
 --output_dir '../ckpt' \
 --output_data_dir '../test_data' \
 --num_train_epochs 3 \
 --dataset 'Amazon' \
 --unseen_combs_path '../data/Amazon/unseen.jsonl' \
 --dataset_path '../data/Amazon/gen.jsonl' \
 --device_num 0 \
 --mode 'Hold-Out' \
 --idx 1 \
 --meta_mctg \
 --lambda_s 0.01
```

## 3. Training Meta MCTG (meta-mctg-sample-training)
When the number of seen attribute combinations is smaller than mini-batch, using basic Meta DCG is not efficient. Meta-mctg-sample-training is a method that control the number of attribute combinations in the train batch to a certain value (i.e., hyperparameter "num_sample_combs") so that we can construct the pseudo-comp batch easier when the number of seen combinations is smaller than mini-batch. For example, in ACD of YELP, the number of seen attribute combinations is 4, which is smaller than mini-batch that equals to 8, if we use common "meta-mctg-training", we will sample a train batch randomly from the dataloader and the number of attribute combinations in the train batch is going to be high probability equal to four (or three whatever). In this casem there will be no other combination to build a pseudo-comp batch.

Note: When the number of seen attribute combinations is smaller than mini-batch, script will enforce the use of meta-mctg sample training. When the number of seen attribute combinations is equal to mini-batch or larger than mini-batch, you can choose using meta-mctg-training or meta-mctg-sample-training. But when you choose using meta-mctg-sample-training, you need to set the value of "num_sample_combs".

In our experience, when the number of seen attribtue combinations is larger than mini-batch (e.g.,Hold-Out/ACD of Fyelp or Hold-Out of Amazon), "meta-mctg-training" produce good enough results. In this situation, we prefer using "meta-mctg-training". However, it is not ruled out that sometimes "meta-mctg-sample-training" will have better results. 

When the number of seen attribute combinations is smaller than mini-batch (e.g., Hold-Out/ACD of Yelp or Hold-Out of Mixture), "meta-mctg-training" almost not work because we can hardly construct the "pseudo-comp batch" based on a ramdom sampled train batch. In this situation, we always use "meta-mctg-sample-training" and we set the value of "num_sample_combs" to 2 for most of the time. It is worth noting that the value of "num_sample_combs" connot be set to 1, as this connot construct the "pseudo-comp batch" either (no compositional generalization when the number of combinations in "train batch" is 1).

```shell
python dcg_meta.py --model_name_or_path 'gpt2-medium' \
 --output_dir '../ckpt' \
 --output_data_dir '../test_data' \
 --num_train_epochs 3 \
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

## 4. CompMCTG Benchmark
For each dataset in CompMCTG Benchmark, the results for each protocol are derived form the average of multiple experiments. The number of partitioning methods included in different protocol across four datasets is presented below:
|Dataset|Original|Hold-Out|ACD|Few-Shot|
|:-|:-:|:-:|:-:|:-:|
|Fyelp|1|40|10|2|
|Amazon|1|12|-|10|
|YELP|1|8|10|8|
|Mixture|1|8|-|8|

Take dataset YELP as an example
- Results of Original
```shell
python dcg_meta.py --dataset 'YELP' --mode 'Original' --idx 0
```
- Average Results of Hold-Out
```shell
for idx in 0 1 2 3 4 5 6 7
do
 python dcg_meta.py --dataset 'YELP' --mode 'Hold-Out' --idx ${idx}
done
```
- Average Results of ACD
```shell
for idx in 0 1 2 3 4 5 6 7 8 9
do
 python dcg_meta.py --dataset 'YELP' --mode 'ACD' --idx ${idx}
done
```
- Average Results of Few-Shot
```shell
for idx in 0 1 2 3 4 5 6 7
do
 python dcg_meta.py --dataset 'YELP' --mode 'Few-Shot' --idx ${idx}
done
```

Similarly, for the dataset Fyelp, Amazon, and Mixture, note that **the results of ACD protocol of dataset Mixture and Amazon is the considered the same as their Few-Shot protocol's counterpart, which implies that the results from this section will be reused in both the Benchmark main table and the Few-Shot results table.**

Then we get the results as follows:
<table>
    <tr>
        <th rowspan="2">Method</th>
        <th colspan="3">Original</th>
        <th colspan="3">Hold-Out</th>
        <th colspan="3">ACD</th>
    </tr>
    <tr>
        <th>ΔA(%)</th>
        <th>P(%)</th>
        <th>P_full(%)</th>
        <th>ΔA(%)</th>
        <th>P(%)</th>
        <th>P_full(%)</th>
        <th>ΔA(%)</th>
        <th>P(%)</th>
        <th>P_full(%)</th>
    </tr>
    <tr>
        <td>LAMA-in-context Learning</td>
        <td>61.35%</td>
        <td>77.30%</td>
        <td>62.16%</td>
        <td>25.55%</td>
        <td>40.82%</td>
        <td>38.50%</td>
        <td>62.93%</td>
        <td>21.17%</td>
        <td>24.63%</td>
    </tr>
    <tr>
        <td>LAMA-2 (Punyakanok et al., 2023)</td>
        <td>57.51%</td>
        <td>57.15%</td>
        <td>56.62%</td>
        <td>18.29%</td>
        <td>49.21%</td>
        <td>18.49%</td>
        <td>57.13%</td>
        <td>49.75%</td>
        <td>18.22%</td>
    </tr>
    <tr>
        <td>ChatGPT (OpenAI, 2023)</td>
        <td>57.51%</td>
        <td>57.15%</td>
        <td>56.62%</td>
        <td>18.29%</td>
        <td>49.21%</td>
        <td>18.49%</td>
        <td>57.13%</td>
        <td>49.75%</td>
        <td>18.22%</td>
    </tr>
</table>






