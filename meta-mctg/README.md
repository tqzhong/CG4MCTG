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
For each dataset in CompMCTG Benchmark, the results for each protocol are derived form the average of multiple experiments, **which means that we will merge all result files for each protocol (categorized as seen and unseen) and input the merged files (merge_seen.jsonl and merge_unseen.jsonl) into our evaluation system to obtain the corresponding results for each protocol**. The number of partitioning methods included in different protocol across four datasets is presented below:
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
        <th rowspan="2">Dataset</th>
        <th colspan="2">Original</th>
        <th colspan="4">Hold-Out</th>
        <th colspan="4">ACD</th>
    </tr>
    <tr>
        <th>A<sub>i.d.</sub></th>
        <th>P<sub>i.d.</sub></th>
        <th>A<sub>i.d.</sub></th>
        <th>P<sub>i.d.</sub></th>
        <th>A<sub>comp</sub></th>
        <th>P<sub>comp</sub></th>
        <th>A<sub>i.d.</sub></th>
        <th>P<sub>i.d.</sub></th>
        <th>A<sub>comp</sub></th>
        <th>A<sub>comp</sub></th>
    </tr>
    <tr>
        <td>Fyelp</td>
        <td>66.43%</td>
        <td>53.31</td>
        <td>66.49%</td>
        <td>53.50</td>
        <td>66.39%</td>
        <td>53.52</td>
        <td>66.01%</td>
        <td>53.29</td>
        <td>64.71%</td>
        <td>53.67</td>
    </tr>
    <tr>
        <td>Amazon</td>
        <td>84.48%</td>
        <td>46.66</td>
        <td>84.71%</td>
        <td>47.20</td>
        <td>84.51%</td>
        <td>47.09</td>
        <td>84.15%</td>
        <td>48.05</td>
        <td>68.28%</td>
        <td>48.36</td>
    </tr>
    <tr>
        <td>YELP</td>
        <td>84.46%</td>
        <td>57.08</td>
        <td>83.07%</td>
        <td>79.05</td>
        <td>80.29%</td>
        <td>80.58</td>
        <td>81.01%</td>
        <td>79.86</td>
        <td>76.08%</td>
        <td>84.30</td>
    </tr>
    <tr>
        <td>Mixture</td>
        <td>84.34%</td>
        <td>68.44</td>
        <td>84.61%</td>
        <td>68.45</td>
        <td>75.45%</td>
        <td>76.41</td>
        <td>83.43%</td>
        <td>57.87</td>
        <td>62.09%</td>
        <td>60.33</td>
    </tr>
</table>

Ultimately, by averaging the results across the four datasets, we obtain the results presented in the main table (Table 1 in the paper) of the CompMCTG Benchmark:
<table>
 <tr>
        <th rowspan="2">Method</th>
        <th colspan="2">Original</th>
        <th colspan="4">Hold-Out</th>
        <th colspan="4">ACD</th>
        <th colspan="3">Average</th>
 </tr>
 <tr>
        <th>A<sub>i.d.</sub></th>
        <th>P<sub>i.d.</sub></th>
        <th>A<sub>i.d.</sub></th>
        <th>P<sub>i.d.</sub></th>
        <th>A<sub>comp</sub></th>
        <th>P<sub>comp</sub></th>
        <th>A<sub>i.d.</sub></th>
        <th>P<sub>i.d.</sub></th>
        <th>A<sub>comp</sub></th>
        <th>A<sub>comp</sub></th>
        <th>A<sub>avg</sub></th>
        <th>P<sub>avg</sub></th>
        <th>G<sub>avg</sub></th>
 </tr>
 <tr>
        <td>DCG</td>
        <td>79.93%</td>
        <td>56.37</td>
        <td>79.72%</td>
        <td>62.05</td>
        <td>76.66%</td>
        <td>64.40</td>
        <td>78.43%</td>
        <td>57.97</td>
        <td>67.70%</td>
        <td>61.11</td>
        <td>76.49%</td>
        <td>60.38</td>
        <td>8.76%</td>
 </tr>
</table>





