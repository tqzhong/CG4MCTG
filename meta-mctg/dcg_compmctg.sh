
# How to use our CompMCTG benchmark?
# There are four datasets in our benchmark: Fyelp, Amazon, Yelp, and Mixture. Each dataset has three or four protocols and each protocol may contain several splits.
# For dataset Fyelp, there are four protocols: Original, Hold-Out, ACD, and Few-Shot
# For dataset Amazon, there are three protocols: Original, Hold-Out, and Few-Shot
# For dataset Yelp, there are four protocols: Original, Hold-Out, ACD, and Few-Shot
# For dataset Mixture, there are three protocols: Original, Hold-Out, and Few-Shot
# The data in the main table in the paper has three main columns: Original-mean, Hold-Out-mean, and ACD-mean and the data in these three columns is the average of the results of the corresponding protocol in the four datasets. For the Amazon and Mixture dataset, their Few-Shot is equivalent to ACD, so when calculating the mean of ACD, the Few-Shot results of these two datasets need to be brought into the calculation.
# In conclusion, Original-mean = (Original-Fyelp + Original-Amazon + Original-Yelp + Original-Mixture) / 4
#                Hold-Out-mean = (Hold-Out-Fyelp + Hold-Out-Amazon + Hold-Out-Yelp + Hold-Out-Mixture) / 4, which contains Hold-Out-mean-i.d. and Hold-Out-mean-comp.
#                ACD-mean = (ACD-Fyelp + Few-Shot-Amazon + ACD-Yelp + Few-Shot-Mixture) / 4, which contains ACD-mean-i.d. and ACD-mean-comp.
#                Average = (Original-mean + Hold-Out-mean-i.d. + Hold-Out-mean-comp. + ACD-mean-i.d. + ACD-mean-comp.) / 5
# After introducing the content above, now we will explain how the results of the four protocols of each dataset are obtained.
# First of all, the Original protocol of each dataset only have one split as it represents the full amount of data. As for the Hold-Out protocol, we choose to traverse all cases where the holdout number is 1 (k=1). Therefore for each dataset, their number of Hold-Out protocol splits is equal to their number of total attribute combinations. (40 splits in Fyelp, 12 splits in Amazon, 8 splits in Yelp, and 8 splits in Mixture). For the ACD protocol, we adopt a greedy-based hill climbing algorithm to sample satisfactory splits which maximize the divergence and finally we get 10 ACD splits in Fyelp and 10 ACD splits in Yelp. For the Few-Shot protocol, we also use the algorithm above to maxmize the divergence but the ratio of the number of combinations in I.D. set and Comp. set is no longer 50% (in ACD, the ratio is 50%) and we get 2 splits in Fyelp, 10 splits in Amazon, 8 splits in Yelp, and 8 splits in Mixture.
# In general, for the results of each protocol in each dataset, you need to use your model to run the experiment on all the splits on the protocol and merge the generated data (by I.D. and Comp.) and merge the results, which will be evaluated at last.
# We use "dcg" as an example to show how to use the CompMCTG benchmark to evaluate this method.

# ================================================================================
# dataset Fyelp
# Original
python dcg_meta.py --dataset Fyelp \
                   --mode Original \
                   --idx 0 \
# Hold-Out
for idx in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39
do
    python dcg_meta.py --dataset Fyelp \
                    --mode Hold-Out \
                    --idx ${idx} \
done
# ACD
for idx in 0 1 2 3 4 5 6 7 8 9
do
    python dcg_meta.py --dataset Fyelp \
                    --mode ACD \
                    --idx ${idx} \
done


# ================================================================================
# dataset Amazon
# Original
python dcg_meta.py --dataset Amazon \
                   --mode Original \
                   --idx 0 \
# Hold-Out
for idx in 0 1 2 3 4 5 6 7 8 9 10 11
do
    python dcg_meta.py --dataset Amazon \
                    --mode Hold-Out \
                    --idx ${idx} \
done
# ACD (equals Few-Shot in Amazon)
for idx in 0 1 2 3 4 5 6 7 8 9
do
    python dcg_meta.py --dataset Amazon \
                    --mode Few-Shot \
                    --idx ${idx} \
done


# ================================================================================
# dataset Yelp
# Original
python dcg_meta.py --dataset Yelp \
                   --mode Original \
                   --idx 0 \
# Hold-Out
for idx in 0 1 2 3 4 5 6 7
do
    python dcg_meta.py --dataset Yelp \
                    --mode Hold-Out \
                    --idx ${idx} \
done
# ACD
for idx in 0 1 2 3 4 5 6 7 8 9
do
    python dcg_meta.py --dataset Yelp \
                    --mode ACD \
                    --idx ${idx} \
done


# ================================================================================
# dataset Mixture
# Original
python dcg_meta.py --dataset Mixture \
                   --mode Original \
                   --idx 0 \
# Hold-Out
for idx in 0 1 2 3 4 5 6 7
do
    python dcg_meta.py --dataset Mixture \
                    --mode Hold-Out \
                    --idx ${idx} \
done
# ACD (equals Few-Shot in Mixture)
for idx in 0 1 2 3 4 5 6 7
do
    python dcg_meta.py --dataset Mixture \
                    --mode Few-Shot \
                    --idx ${idx} \
done