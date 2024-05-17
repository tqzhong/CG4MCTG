## Datasets of CompMCTG Benchmark

Please click [here](https://drive.google.com/file/d/1qUI7u2F9gYweinAtz3bGB7B7hpkSAOby/view?usp=sharing) to download the data.

```shell
mv data_compmctg.zip ./data
cd ./data
unzip data_compmctg.zip
```

### 1. Contents

```
├── Amazon
│   ├── cls.jsonl
│   ├── gen.jsonl
│   └── unseen.jsonl
├── Fyelp
│   ├── cls.jsonl
│   ├── gen.jsonl
│   ├── unseen.jsonl
├── Mixture
│   ├── cls.jsonl
│   ├── gen.jsonl
│   ├── unseen.jsonl
├── Yelp
│   ├── cls.jsonl
│   ├── gen.jsonl
│   ├── unseen.jsonl
```

In this path, there are four folders named "Amazon", "Fyelp", "Mixture", and "Yelp". Each folder contains three jsonl files: "cls.jsonl", which is used for training the classifier; "gen.jsonl", which is used for training the generative model; and "unseen.jsonl", which includes all compositional attribute combinations under all modes of this dataset.

### 2. Datasets Information

#### 2.1 Fyelp
4 attributes: "sentiment", "gender", "cuisine", "tense"  
		"sentiment"&isin;{"Positive","Negative"}  
		"gender"&isin;{"Male","Female"}  
		"cuisine"&isin;{"Asian", "American", "Mexican", "Bar", "dessert"}  
		"tense"&isin;{"Present","Past"}  

attribute combinations: 40
		classifier data number: 40000, 1000 per one combination
		generation data number: 70000, 1750 per one combination

#### 2.2 Amazon

2 attributes: "sentiment", "topic" 
		"sentiment"&isin;{"Positive","Negative"}  
		"topic"&isin;{"Books", "Clothing", "Music", "Electronics", "Movies", "Sports"}

attribute combinations: 12
		classifier data number: 180000, 15000 per one combination
		generation data number: 120000, 10000 per one combination

#### 2.3 Yelp
3 attributes: "sentiment", "pronoun", "tense" 
		"sentiment"&isin;{"Positive","Negative"}  
		"pronoun"&isin;{"plural","singular"}
		"tense"&isin;{"Present","Past"}

attribute combinations: 8
		classifier data number: 24000, 3000 per one combination
		generation data number: 24000, 3000 per one combination

#### 2.4 Mixture(IMDB, OpeNER and Sentube)
2 attributes: "sentiment", "topic_cged" 
		"sentiment"&isin;{"Positive","Negative"}  
		"topic_cged"&isin;{"movies", "opener", "tablets", "auto"}

attribute combinations: 8
		classifier data number: 4264, 533 per one combination
		generation data number: 4800, 600 per one combination

### 3. Usage about the datasets

please check ./data/utils.py to get the usage of the datasets
