# SememePSO-Attack
Code and data of the ACL 2020 paper "Word-level Textual Adversarial Attacking as Combinatorial Optimization". [[paper]](https://arxiv.org/pdf/1910.12196.pdf)
## Citation
Please cite our paper if you find it helpful.


```
@inproceedings{zang2020word,
  title={Word-level Textual Adversarial Attacking as Combinatorial Optimization},
  author={Zang, Yuan and Qi, Fanchao and Yang, Chenghao and Liu, Zhiyuan and Zhang, Meng and Liu, Qun and Sun, Maosong},
  booktitle={Proceedings of ACL},
  year={2020}
}
```
This repository is mainly contributed by Yuan Zang and Chenghao Yang.
## Requirements

- tensorflow-gpu == 1.14.0   
- keras == 2.2.4   
- sklearn == 0.0  
- anytree == 2.6.0  
- nltk == 3.4.5  
- OpenHowNet == 0.0.1a8    
- pytorch_transformers == 1.0.0  
- loguru == 0.3.2
## General Required Data and Tools
- Download [Glove vectors](http://nlp.stanford.edu/data/glove.840B.300d.zip)
<!-- ### Download Stanford Pos Tagger -->
- Download [Stanford POS Tagger](https://nlp.stanford.edu/software/tagger.shtml#Download)

## Reproducibility Support
Since data processing and models training may take a lot of time and computing resources, we provide the data and models we use for experiments. You can directly download the data and models we used for related experiments from [TsinghuaCloud](https://cloud.tsinghua.edu.cn/d/b6b35b7b7fdb43c1bf8c/) or [Google Drive](https://drive.google.com/drive/folders/1EiF6tYhqGRjXeIG7r0NdJMUT3ksb_gnL?usp=sharing). The instructions of how to use these files can be found in the `README.md` files in `IMDB/`, `SNLI/` and `SST/`.

## Running Instructions
Please see the `README.md` files in `IMDB/`, `SNLI/` and `SST/` for specific running instructions for each attack models on corresponding downstream tasks.