import json
import torch
from torch import nn, optim
import random
from loguru import logger
from pytorch_transformers import BertTokenizer, BertModel, AdamW, WarmupLinearSchedule

from IMDBdata.IMDBconfig import filenames, parameters
import os
if not os.path.exists('log'):
    os.mkdir('log')
tokenizer = BertTokenizer.from_pretrained(parameters.BERT_MODEL)

def getTime():
    import time
    return str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))



def adjustBatchInputLen(batch):
    inputs = batch["inputs"]
    length = 0
    for item in inputs:
        length = max(length, len(item))
    length = min(length, parameters.max_def_lens)
    num = len(inputs)
    for i in range(num):
        if length > len(inputs[i]):
            for j in range(length - len(inputs[i])):
                inputs[i].append(0)
        else:
            inputs[i] = inputs[i][:length]


def prepareBatches(ids, id2content):
    res = []
    random.shuffle(ids)
    # nasari = json.load(open("data/cleaned_nasari.json",'r',encoding='utf-8'))
    for head_id in range(0, len(ids), parameters.batch_size):
        batch_ids = ids[head_id:head_id + parameters.batch_size]
        batch = {}
        batch["inputs"] = []
        batch['labels'] = []
        #batch['token_type_ids'] = []
        # batch['nasari'] = []
        for _id in batch_ids:
            # Merge all definitions:
            # 1) Maybe various defs demonstrate incomplete synset semantics
            # 2) Avoid being biased by the number of defs
            defs = ["[SEP]".join(id2content[_id]["en_defs"])]
            # Do not worry about the warning "token indices sequence length is longer than..." here, becuase adjustInputLen will handle this problem by truncating the input length to acceptible range.
            batch["inputs"].extend([tokenizer.convert_tokens_to_ids(tokenizer.tokenize("[CLS] " + x)) for x in defs])
            #ans_sememes = [label2id[x] for x in id2content[_id]["ch_sememes"]]
            #labels = torch.zeros((len(label2id),)).float()
            batch["labels"].append(label2id[id2content[_id]['label']])
            #token_type_ids = [0]*(len(id2content[_id]['en_defs'][0])+2)+[1]*(parameters.max_def_lens-1-len(id2content[_id]['en_defs'][0]))
            #batch['token_type_ids'].append(token_type_ids)
            # if _id in nasari:
            #     batch['nasari'].append(torch.FloatTensor(nasari[_id]))
            # else:
            #     batch['nasari'].append(torch.zeros((300,)).float())
            # for x in ans_sememes:
            #     labels[x] = 1
            # for i in range(len(defs)):
            #     batch['labels'].append(labels)
        adjustBatchInputLen(batch)
        end_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("[SEP]"))[0]
        for i in range(len(batch["inputs"])):
            batch["inputs"][i].append(end_id)
        batch["inputs"] = torch.stack([torch.LongTensor(x) for x in batch['inputs']])
        #batch["token_type_ids"] = torch.stack([torch.LongTensor(x) for x in batch['token_type_ids']])
        batch['labels'] = torch.Tensor(batch['labels']).view((len(batch['labels'],))).long()
        # batch['nasari'] = torch.stack(batch['nasari'])
        res.append(batch)
    # del nasari
    return res

    # for i in range(len(defs)):
    #     defs[i] = "[CLS] %s [SEP]" % defs[i]

def map_criterion(logits,labels):
    res_map = 0
    # for logit, label in zip(logits,labels):
    #     local_map = 0
    #     ans_list = set()
    #     for index, i in enumerate(label):
    #         if i.item() == 1:
    #             ans_list.add(index)
    #     predict_list = list()
    #     for index, i in enumerate(logit):
    #         predict_list.append((index, i))
    #     predict_list.sort(key=lambda x:x[1],reverse=True)
    #     count = 0
    #     for index, (sememe_id, pr) in enumerate(predict_list):
    #         if sememe_id in ans_list:
    #             count += 1
    #             local_map += count / (index + 1)
    #     res_map += local_map / len(ans_list)
    orders = torch.argsort(logits, dim=-1, descending=True).cpu()
    for order, label in zip(orders, labels):
        ans_list = set()
        for index, i in enumerate(label):
            if i.item() == 1:
                ans_list.add(index)
        count = 0
        local_map = 0
        for index, item in enumerate(order):
            if item.item() in ans_list:
                count += 1
                local_map += count / (index + 1)
        res_map += local_map / len(ans_list)


    return res_map

class ModelTrainer(nn.Module):
    def __init__(self, model, output_size):
        super(ModelTrainer, self).__init__()
        self.model = model
        #self.decoder = nn.Linear(300, output_size)
        # self.decoder = nn.Linear(self.model.config.hidden_size+300, output_size)
        self.decoder = nn.Linear(self.model.config.hidden_size, output_size)
        # self.criterion = nn.MultiLabelSoftMarginLoss()
        self.criterion = nn.CrossEntropyLoss()
        self.if_cuda = torch.cuda.is_available()

    def forward(self, batch):
        if self.if_cuda:
            batch["inputs"] = batch["inputs"].cuda()
            batch["labels"] = batch["labels"].cuda()
            #batch['token_type_ids'] = batch["token_type_ids"].cuda()
            # batch["nasari"] = batch["nasari"].cuda()

        #token_type_ids = torch.zeros_like(batch["inputs"])
        # with torch.no_grad():
        outputs = self.model(batch["inputs"])
        encoded_layers = outputs[0]
        encode_out = encoded_layers[:,0,:]
        #encode_out = torch.mean(encoded_layers, dim=1)
        # encode_out = torch.cat([encode_out,batch["nasari"]], dim=-1)
        #encode_out = batch['nasari']
        logits = self.decoder(encode_out)
        loss = self.criterion(logits, batch['labels'])
        return loss, logits
if __name__ == "__main__":
    instance_name = "BERTModel"
    logger.add(open("log/"+instance_name+".txt", 'w', encoding='utf-8'), colorize=True,
               format="{time} {level} {message}", backtrace=True, diagnose=True)

    label2id = json.load(open("IMDBdata/" + filenames.label2id, 'r', encoding='utf-8'))
    random.seed(parameters.random_seed)
    torch.manual_seed(parameters.random_seed)
    torch.cuda.manual_seed(parameters.random_seed)
    logger.info("Parameters:\n",str(parameters))
    train_ids = json.load(open("IMDBdata/"+filenames.train_set, 'r', encoding='utf-8'))
    valid_ids = json.load(open("IMDBdata/"+filenames.valid_set, 'r', encoding='utf-8'))
    test_ids = json.load(open("IMDBdata/"+filenames.test_set, 'r', encoding='utf-8'))
    id2content = json.load(open("IMDBdata/"+filenames.id2content, 'r', encoding='utf-8'))

    train_batches = prepareBatches(train_ids, id2content)
    valid_batches = prepareBatches(valid_ids, id2content)
    test_batches = prepareBatches(test_ids, id2content)

    model = BertModel.from_pretrained(parameters.BERT_MODEL)
    model.eval()

    trainer = ModelTrainer(model, len(label2id))
    if torch.cuda.is_available():
        trainer = trainer.cuda()
        trainer.model = trainer.model.cuda()

    logger.info("Model Arch:\n"+str(trainer))

    optimizer = optim.Adam(trainer.parameters(), lr=2e-5, eps=1e-8)
    #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0, t_total=len(train_ids) * 3.0)


    best_acc = 0
    for i in range(parameters.epoches):
        # train
        acc = 0
        local_loss = 0
        for batches, mode in zip([train_batches, valid_batches], ["train", 'eval']):
            logger.info("Mode: %s" % mode)
            if mode == "eval":
                trainer = trainer.eval()
            else:
                trainer = trainer.train()
            for batch in batches:
                optimizer.zero_grad()
                loss, logits = trainer(batch)
                acc += sum(torch.argmax(logits, dim=-1) == batch['labels']).float() 
                local_loss += loss.item()
                if mode == "train":
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(trainer.parameters(), 1.0)
                    #scheduler.step()
                    optimizer.step()
            if mode == "train":
                acc /= len(train_ids)
                local_loss /= len(train_ids)
            else:
                acc /= len(valid_ids)
                local_loss /= len(valid_ids)
                if acc >= best_acc:
                    best_acc = acc
                    torch.save(trainer.state_dict(), instance_name)
            logger.info("Epoch: {0}, Loss: {1}, Acc: {2}".format(i, local_loss, acc))

    trainer.load_state_dict(torch.load(instance_name))

    acc = 0
    local_loss = 0
    for batch in test_batches:
        loss, logits = trainer(batch)
        #print(torch.argmax(logits, dim=-1), batch['labels'])
        acc += sum(torch.argmax(logits, dim=-1) == batch['labels']).float()
        local_loss += loss.item()

    local_loss /= len(test_ids)
    acc /= len(test_ids)
    logger.info("Mode: %s" % "test")
    logger.info("Loss: {0}, acc: {1}".format(local_loss, acc))


