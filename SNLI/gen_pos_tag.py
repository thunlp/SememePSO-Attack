import pickle

with open('./all_seqs.pkl', 'rb') as fh:
    train,valid,test = pickle.load(fh)
with open('./nli_tokenizer.pkl', 'rb') as fh:
    tokenizer = pickle.load(fh)
dict = {w: i for (w, i) in tokenizer.word_index.items()}
inv_dict = {i: w for (w, i) in dict.items()}
word_candidate={}
trains=[t[1:-1] for t in train['s2']]
tests=[t[1:-1] for t in test['s2']]
from nltk.tag import StanfordPOSTagger
jar = 'stanford-postagger-2018-10-16/stanford-postagger.jar'
model = 'stanford-postagger-2018-10-16/models/english-left3words-distsim.tagger'
pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')

#from nltk.stem import WordNetLemmatizer
#wnl = WordNetLemmatizer()
#pos_tag=pos_tagger.tag(['what', "'s", 'invigorating', 'about', 'it', 'is', 'that', 'it', 'does', "n't", 'give', 'a', 'damn'])
#print(pos_tag)

train_text=[[inv_dict[t] for t in tt] for tt in trains]

all_pos_tags=[]

for text in train_text:

    pos_tags = pos_tagger.tag(text)
    all_pos_tags.append(pos_tags)
f=open('pos_tags.pkl','wb')
pickle.dump(all_pos_tags,f)
f.close()

test_text = [[inv_dict[t] for t in tt] for tt in tests]

all_pos_tags = []

for text in test_text:
    pos_tags = pos_tagger.tag(text)
    all_pos_tags.append(pos_tags)
f = open('pos_tags_test.pkl', 'wb')
pickle.dump(all_pos_tags, f)
f.close()
