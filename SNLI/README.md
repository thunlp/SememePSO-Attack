# README
## Necessary Data Download
### Download SNLI Data
Download SNLI data from https://nlp.stanford.edu/projects/snli/snli_1.0.zip
Download dataset.zip from https://cloud.tsinghua.edu.cn/d/b6b35b7b7fdb43c1bf8c/
### Download Processed SNLI Data for BERT
Download SNLIdata.zip from https://cloud.tsinghua.edu.cn/d/b6b35b7b7fdb43c1bf8c/
## Process Data and Train Models
Since processing data and training models may take a lot of time and computing resources, we provide used data and models.  
You can download them in IMDB_used_data.zip from https://cloud.tsinghua.edu.cn/d/b6b35b7b7fdb43c1bf8c/.  
Use the data and models you can reproduce the results reported in the paper.   
You can also process data and train models by following steps.
### Preprocess SNLI Data
Run tokenize_snli.py
Run preprocess.py
### Generate Candidate Substitution Words
Run gen_pos_tag.py
Run lemma.py
Run gen_candidates.py
### Train BiLSTM Model
Run train_model.py
### Train BERT Model
Run SNLI_BERT.py
## Craft Adversarial Examples
### Crafting Adversarial Examples for Bi-LSTM
Run AD_dpso_sem.py
### Crafting Adversarial Examples for BERT
Run AD_dpso_sem_bert.py
