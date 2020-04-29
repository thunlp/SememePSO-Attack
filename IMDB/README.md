# README
## Requirements
tensorflow-gpu == 1.14.0   
keras == 2.2.4   
sklearn == 0.0  
anytree == 2.6.0  
nltk == 3.4.5
OpenHowNet == 0.0.1a8
pytorch_transformers == 1.0.0
loguru == 0.3.2
## Necessary Data Download
### Download IMDB Data
Download IMDB data from http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
### Download Processed IMDB Data for BERT
Download IMDBdata.zip from https://cloud.tsinghua.edu.cn/d/b6b35b7b7fdb43c1bf8c/
## Process Data and Train Models
Since data processing and models training may take a lot of time and computing resources, we provide used data and models.  
You can download them in IMDB_used_data.zip from https://cloud.tsinghua.edu.cn/d/b6b35b7b7fdb43c1bf8c/.  
Use the data and models you can reproduce the results reported in the paper.   
You can also process data and train models by following steps.
### Process IMDB Data
Run build_embeddings.py
### Generate Candidate Substitution Words
Run gen_pos_tag.py
Run lemma.py
Run gen_candidates.py
### Train BiLSTM Model
Run train_model.py
### Train BERT Model
Run IMDB_BERT.py
## Craft Adversarial Examples
### Craft Adversarial Examples for Bi-LSTM
Run AD_dpso_sem.py

### Craft Adversarial Examples for BERT
Run AD_dpso_sem_bert.py

