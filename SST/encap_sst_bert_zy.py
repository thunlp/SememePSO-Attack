from SST_BERT import ModelTrainer
from SST_BERT import adjustBatchInputLen
from pytorch_transformers import BertTokenizer, BertModel, AdamW, WarmupLinearSchedule
from torch import nn
import torch
import config
class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.config = config.SSTConfig()
        model = BertModel.from_pretrained(self.config.BERT_MODEL)
        self.model = ModelTrainer(model, 2)
        self.model.load_state_dict(torch.load(self.config.model_name))
        self.model.eval()
        self.dataset = dataset
        self.tokenizer = BertTokenizer.from_pretrained(self.config.BERT_MODEL)


    def forward(self,input_x):
        num_instance = len(input_x)
        batch = dict()
        batch["inputs"] = []
        batch["labels"] = torch.zeros((num_instance,)).long()
        for i in range(len(input_x)):
            tokens = list()
            tokens.append(self.tokenizer.cls_token)
            for j in range(len(input_x[i])):
                #print(input_x[i], tokens)
                #print(type(input_x[i][j]))
                #print(self.dataset.inv_dict[0])
                # inv_dict has no padding, maybe because of keras setting
                if input_x[i][j] != 0:
                    tokens.append(self.dataset.inv_dict[int(input_x[i][j])])
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            batch["inputs"].append(tokens)
        adjustBatchInputLen(batch)
        end_id = self.tokenizer.convert_tokens_to_ids("[SEP]")
        for i in range(len(input_x)):
            tokens = batch["inputs"][i]
            tokens.append(end_id)
        batch["inputs"] = torch.stack([torch.LongTensor(x) for x in batch['inputs']])
        with torch.no_grad():
            loss, logits = self.model(batch)
        return logits.cpu().numpy()

    def predict(self, input_x):
        return self(input_x)



    def adjustBatchInputLen(self, batch):
        inputs = batch["inputs"]
        length = 0
        for item in inputs:
            length = max(length, len(item))
        length = min(length, self.config.max_sent_lens)
        num = len(inputs)
        for i in range(num):
            if length > len(inputs[i]):
                for j in range(length - len(inputs[i])):
                    inputs[i].append(self.tokenizer.pad_token_id)
            else:
                inputs[i] = inputs[i][:length]

