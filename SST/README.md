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
1. Download Processed SST-2 Data  
Download sst-2.zip from https://cloud.tsinghua.edu.cn/d/b6b35b7b7fdb43c1bf8c/
2. Download Glove Vectors  
Download glove vectors from http://nlp.stanford.edu/data/glove.840B.300d.zip
3. Download Stanford Pos Tagger  
Download Stanford Pos Tagger from https://nlp.stanford.edu/software/tagger.shtml#Download
4. Download Processed SST-2 Data for BERT  
Download SST2data.zip from https://cloud.tsinghua.edu.cn/d/b6b35b7b7fdb43c1bf8c/
## Process Data and Train Models
Since processing data and training models may take a lot of time and computing resources, we provide used data and models.  
You can download them in IMDB_used_data.zip from https://cloud.tsinghua.edu.cn/d/b6b35b7b7fdb43c1bf8c/.  
Use the data and models you can reproduce the results reported in the paper.   
You can also process data and train models by following steps.
#### Process SST-2 Data  
Run data_utils.py
#### Generate Candidate Substitution Words  
Run gen_pos.py
Run lemma.py
Run gen_candidates.py
#### Train BiLSTM Model  
Run train_model.py
#### Train BERT Model  
Run SST_BERT.py
## Craft Adversarial Examples
#### Crafting Adversarial Examples  
Run AD_dpso_sem.py
#### Crafting Adversarial Examples  
Run AD_dpso_sem_bert.py
