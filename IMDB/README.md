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
### Download Glove Vectors
Download glove vectors from http://nlp.stanford.edu/data/glove.840B.300d.zip

### Download Stanford Pos Tagger
Download Stanford Pos Tagger from https://nlp.stanford.edu/software/tagger.shtml#Download
## Data Preprocess
### Preprocess IMDB Data
Run build_embeddings.py
### Generate Candidate Substitution Words
Run gen_pos_tag.py
Run lemma.py
Run gen_candidates.py
## Train BiLSTM Model
Run train_model.py
## Crafting Adversarial Examples
Run AD_dpso_sem.py
## Train BERT Model
Run IMDB_BERT.py
## Crafting Adversarial Examples
Run AD_dpso_sem_bert.py