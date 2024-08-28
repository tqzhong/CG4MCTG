## Evaluation of CompMCTG Benchmark

Please click [here](https://huggingface.co/Tianqi-Zhong/Classifiers_For_CompMCTG) to download the models for evaluation.

```shell
mv classifiers.zip ./evaluation
cd ./evaluation
unzip classifiers.zip
```

### 1. Contents

```shell
├── classifiers
│   ├── Amazon
│       ├── sentiment
│       ├── topic
│   ├── Fyelp
│       ├── sentiment
│       ├── cuisine
│       ├── gender
│       ├── tense
│   ├── Mixture
│       ├── sentiment
│       ├── topic_cged
│   └── Yelp
│       ├── sentiment
│       ├── pronoun
│       ├── tense
├── scripts
│   ├── eval_acc_Amazon.py
│   ├── eval_acc_Fyelp.py
│   ├── eval_acc_Mixture.py
│   ├── eval_acc_Yelp.py
│   ├── eval_compmctg.py
│   ├── eval_perplexity.py
│   └── model.py
├── test_cases
│   ├── cases_Amazon.jsonl
│   ├── cases_Fyelp.jsonl
│   ├── cases_Mixture.jsonl
│   └── cases_Yelp.jsonl
└── eval_compmctg.sh
```

In this path, there are four folders named "Amazon", "Fyelp", "Mixture",  and "Yelp". Each folder contains several classifiers.

### 2. Evaluation

```shell
bash eval_compmctg.sh
```

**Parameters:**

- --dataset_path: path of the file you want to evaluate.
- --dataset: choices = ['Fyelp', 'Amazon', 'Yelp', 'Mixture'], the dataset category of the test file.
- --device_num: GPU id.

### 3. Test Cases

All files applied to our CompMCTG Benchmark test must meet the following format requirements: it is a jsonl file, and each piece of data contains the "text" key and the corresponding attribute key. For test sample files, check the test_cases.
