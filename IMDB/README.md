# Running instructions on IMDB dataset
## Data
- Download official IMDB data from [here](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)
- Download processed IMDB data for training models: [IMDBdata.zip](https://cloud.tsinghua.edu.cn/d/b6b35b7b7fdb43c1bf8c/files/?p=%2FIMDBdata.zip)




## Process Data and Train Model
If you do not want to use our processed data and models, you can also process data and train models by following steps.
- Process IMDB Data
```bash
python build_embeddings.py
```
- Generate Candidate Substitution Words
```bash
python gen_pos_tag.py
python lemma.py
python gen_candidates.py
```
- Train BiLSTM Model
```bash
python train_model.py
```
- Train BERT Model
```bash
python IMDB_BERT.py
```
## Craft Adversarial Examples
- Craft Adversarial Examples for Bi-LSTM
```bash
python AD_dpso_sem.py
```
- Craft Adversarial Examples for BERT
```bash
python AD_dpso_sem_bert.py
```

