import numpy as np
from data import get_nli, get_batch, build_vocab
train, valid, test = get_nli('dataset/SNLI/')
print(train['label'][:3])
print(train['s1'][:3])
import pickle

tokenizer=pickle.load(open('nli_tokenizer.pkl','rb'))
vocab = {w: i for (w, i) in tokenizer.word_index.items()}
inv_vocab={i: w for (w, i) in tokenizer.word_index.items()}


word_vec = build_vocab(train['s1'] + train['s2'] +
                       valid['s1'] + valid['s2'] +
                       test['s1'] + test['s2'], 'glove.840B.300d.txt')
new_word_vec={}
new_word_vec['<s>']=word_vec['<s>']
new_word_vec['</s>']=word_vec['</s>']
new_word_vec['<p>']=word_vec['<p>']
new_word_vec[0]=word_vec['<p>']


with open('glove.840B.300d.txt') as f:
    for line in f:
        word, vec = line.split(' ', 1)
        if word in vocab:
            new_word_vec[vocab[word]] = np.array(list(map(float, vec.split())))
        if word=='UNK':
            glove_unk=np.array(list(map(float, vec.split())))
with open('glove_unk.pkl','wb') as fw:
    pickle.dump(glove_unk,fw)

print('Found {0}(/{1}) words with glove vectors'.format(
            len(word_vec), len(vocab)))


def r(train):
    t=[]
    for i in range(len(train)):
        p=[]
        for x in train[i]:
            if x==0:
                break
            p.append(x)
        t.append(p)
    return t
f=open('word_vec.pkl','wb')
pickle.dump(new_word_vec,f)

train_orig,valid_orig,test_orig=pickle.load(open('nli_seqs.pkl','rb'))
print('visual_train:',train_orig[0][0],train_orig[0][1])
new_train={}
new_valid={}
new_test={}
new_train['s1']=r(train_orig[0])
new_train['s2']=r(train_orig[1])
new_train['label']=[np.argmax(t) for t in train_orig[2]]
new_valid['s1']=r(valid_orig[0])
new_valid['s2']=r(valid_orig[1])
new_valid['label']=[np.argmax(t) for t in valid_orig[2]]
new_test['s1']=r(test_orig[0])
new_test['s2']=r(test_orig[1])
new_test['label']=[np.argmax(t) for t in test_orig[2]]
for split in ['s1', 's2']:
    for data_type in ['new_train', 'new_valid', 'new_test']:
        eval(data_type)[split] = np.array([['<s>'] +
            [word for word in sent if word in new_word_vec] +
            ['</s>'] for sent in eval(data_type)[split]])
f=open('all_seqs.pkl','wb')
pickle.dump((new_train,new_valid,new_test),f)
