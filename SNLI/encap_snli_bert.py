from SNLI_BERT import ModelTrainer
from SNLI_BERT import adjustBatchInputLen
from pytorch_transformers import BertTokenizer, BertModel, AdamW, WarmupLinearSchedule
from torch import nn
import torch
import config
class Model(nn.Module):
    def __init__(self, inv_dict):
        super(Model, self).__init__()
        self.config = config.SNLIConfig()
        model = BertModel.from_pretrained(self.config.BERT_MODEL)
        self.model = ModelTrainer(model, 3)
        self.model.load_state_dict(torch.load(self.config.model_name))
        self.model = self.model.eval().cuda()
        self.inv_dict = inv_dict
        self.tokenizer = BertTokenizer.from_pretrained(self.config.BERT_MODEL)
        self.m = nn.Softmax(1)


    def forward(self,input_x):
        assert len(input_x[0]) == len(input_x[1]), "premise and hypothesis should share the same batch lens!"
        num_instance = len(input_x[0])
        batch = dict()
        batch["inputs"] = []
        batch["labels"] = torch.zeros((num_instance,)).long()
        for i in range(len(input_x[0])):
            tokens = list()
            tokens.append(self.tokenizer.cls_token)
            for k in [0, 1]:
                add_sep = False
                if k == 0:
                    add_sep = True
                for j in range(len(input_x[k][i])):
                    #print(input_x[i], tokens)
                    #print(type(input_x[i][j]))
                    #print(self.dataset.inv_dict[0])
                    # inv_dict has no padding, maybe because of keras setting
                    if input_x[k][i][j] != 0:
                        tokens.append(self.inv_dict[int(input_x[k][i][j])])
                if add_sep:
                    tokens.append("[SEP]")
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            batch["inputs"].append(tokens)
        adjustBatchInputLen(batch)
        end_id = self.tokenizer.convert_tokens_to_ids("[SEP]")
        for i in range(len(input_x[0])):
            tokens = batch["inputs"][i]
            tokens.append(end_id)
        batch["inputs"] = torch.stack([torch.LongTensor(x) for x in batch['inputs']])
        with torch.no_grad():
            loss, logits = self.model(batch)
        logits = self.m(logits[:,[1,0,2]])
        return logits.cpu().numpy()

    def predict(self, input_x):
        # sess is of no use, just to tailor the ugly interface
        return self(input_x)

    def pred(self, x, y):
        return self([x, y])



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

