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
### Download SNLI Data
Download SNLI data from https://nlp.stanford.edu/projects/snli/snli_1.0.zip
Download dataset.zip from https://cloud.tsinghua.edu.cn/d/b6b35b7b7fdb43c1bf8c/
### Download Glove Vectors
Download glove vectors from http://nlp.stanford.edu/data/glove.840B.300d.zip
### Download Stanford Pos Tagger
Download Stanford Pos Tagger from https://nlp.stanford.edu/software/tagger.shtml#Download
### Download Processed SNLI Data for BERT
Download SNLIdata.zip from https://cloud.tsinghua.edu.cn/d/b6b35b7b7fdb43c1bf8c/
## Data Preprocess
### Preprocess SNLI Data
Run tokenize.py
Run preprocess.py
### Generate Candidate Substitution Words
Run gen_pos_tag.py
Run lemma.py
Run gen_candidates.py
## Train BiLSTM Model
Run train_model.py
## Crafting Adversarial Examples
Run AD_dpso_sem.py
## Train BERT Model
Run SNLI_BERT.py
## Crafting Adversarial Examples
Run AD_dpso_sem_bert.py