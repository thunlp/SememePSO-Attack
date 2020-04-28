class DianpingConfig:
    def __init__(self):
        self.instance_name = "BERTModel.pt"
        self.model_name = self.instance_name 
        self.BERT_MODEL = "bert-base-chinese"
        self.max_sent_lens = 64
class SSTConfig:
    def __init__(self):
        self.instance_name = "BERTModel.pt"
        self.model_name = self.instance_name 
        self.BERT_MODEL = "bert-base-uncased"
        self.max_sent_lens = 32
class SNLIConfig:
    def __init__(self):
        self.instance_name = "BERTModel.pt"
        self.model_name = self.instance_name 
        self.BERT_MODEL = "bert-base-uncased"
        self.max_sent_lens = 64
class IMDBConfig:
    def __init__(self):
        self.instance_name = "BERTModel.pt"
        self.model_name = self.instance_name 
        self.BERT_MODEL = "bert-base-uncased"
        self.max_sent_lens = 254
class LCQMCConfig:
    def __init__(self):
        self.instance_name = "BERTModel.pt"
        self.model_name = self.instance_name 
        self.BERT_MODEL = "bert-base-chinese"
        self.max_sent_lens = 64
