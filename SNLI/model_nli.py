import pickle
import numpy as np

import os

from torch.autograd import Variable
import torch
import torch.nn as nn
from infersent_model import NLINet
class Model():
    def __init__(self):
        self.word_vec = pickle.load(open('word_vec.pkl', 'rb'))
        self.word_vec[42391]=pickle.load(open('glove_unk.pkl','rb'))
        #print(self.word_vec[42391])
        config_nli_model = {
            'n_words': len(self.word_vec),
            'word_emb_dim': 300,
            'enc_lstm_dim': 2048,
            'n_enc_layers': 1,
            'dpout_model': 0.,
            'dpout_fc': 0.,
            'fc_dim': 512,
            'bsize': 64,
            'n_classes': 3,
            'pool_type': 'max',
            'nonlinear_fc': 1,
            'encoder_type': 'InferSent',
            'use_cuda': True,
        }
        self.nli_net = NLINet(config_nli_model)
        self.nli_net.load_state_dict(torch.load(os.path.join('savedir/', 'model_nli.pickle'), map_location='cuda:0'))
        self.nli_net.encoder.load_state_dict(
            torch.load(os.path.join('savedir/', 'model_nli.pickle' + '.encoder.pkl'), map_location='cuda:0'))

        self.nli_net.cuda()
        self.word_emb_dim=300
    def get_batch(self,batch, word_vec, emb_dim=300):
        # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
        lengths = np.array([len(x) for x in batch])
        max_len = np.max(lengths)
        embed = np.zeros((max_len, len(batch), emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = word_vec[batch[i][j]]
        return torch.from_numpy(embed).float(), lengths
    def evaluate(self,s1, s2, target):
        self.nli_net.eval()
        correct = 0.
        batch_size = 64
        word_emb_dim = 300


        for i in range(0, len(s1), batch_size):
            # prepare batch
            s1_batch, s1_len = self.get_batch(s1[i:i + batch_size], self.word_vec, word_emb_dim)
            s2_batch, s2_len = self.get_batch(s2[i:i + batch_size], self.word_vec, word_emb_dim)
            s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
            tgt_batch = Variable(torch.LongTensor(target[i:i + batch_size])).cuda()

            # model forward
            output = self.nli_net((s1_batch, s1_len), (s2_batch, s2_len))

            pred = output.data.max(1)[1]
            correct += pred.long().eq(tgt_batch.data.long()).cpu().sum().item()

            # save model
        eval_acc = 100 * float(correct) / len(s1)
        print('finalgrep : accuracy test :', eval_acc)

    def softmax_pred(self,n):
        s = [np.exp(t) for t in n]
        ss = np.sum(s)
        result = [t / ss for t in s]
        return result

    def pred(self,s1, s2):
        s1=[['<s>']+s+['</s>'] for s in s1]
        s2 = [['<s>'] + s + ['</s>'] for s in s2]
        s1_batch, s1_len = self.get_batch(s1, self.word_vec, self.word_emb_dim)
        s2_batch, s2_len = self.get_batch(s2, self.word_vec, self.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        output = self.nli_net((s1_batch, s1_len), (s2_batch, s2_len))
        pred = output.data.cpu().numpy()
        new_pred=[]
        for p in pred:
            new_pred.append(self.softmax_pred(p))
        return np.array(new_pred)
