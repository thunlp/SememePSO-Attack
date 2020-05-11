# Running instructions on SNLI dataset
## Data
- Download official SNLI data from [here](https://nlp.stanford.edu/projects/snli/snli_1.0.zip)
- Download [dataset.zip](https://cloud.tsinghua.edu.cn/d/b6b35b7b7fdb43c1bf8c/files/?p=%2Fdataset.zip)
- Download processed SNLI data for training models:  [SNLIdata.zip](https://cloud.tsinghua.edu.cn/d/b6b35b7b7fdb43c1bf8c/files/?p=%2FSNLIdata.zip)

## Process Data and Train Model

- Preprocess SNLI Data
```bash
python tokenize_snli.py
python preprocess.py
```
- Generate Candidate Substitution Words
```bash
python gen_pos_tag.py
python lemma.py
python gen_candidates.
```
- Train BiLSTM Model
```bash
python train_model.py
```
- Train BERT Model
```bash
python SNLI_BERT.py
```
## Craft Adversarial Examples
- Crafting Adversarial Examples for Bi-LSTM
```bash
python AD_dpso_sem.py
```
- Crafting Adversarial Examples for BERT
```bash
python AD_dpso_sem_bert.py
```
