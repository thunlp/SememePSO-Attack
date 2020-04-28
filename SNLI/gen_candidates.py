import pickle

#f=open('en_aux_files/dataset_13837.pkl','rb')
#dataset=pickle.load(f)
word_candidate={}


with open('./all_seqs.pkl', 'rb') as fh:
    train,valid,test = pickle.load(fh)
with open('./nli_tokenizer.pkl', 'rb') as fh:
    tokenizer = pickle.load(fh)
vocab = {w: i for (w, i) in tokenizer.word_index.items()}
inv_dict = {i: w for (w, i) in vocab.items()}
word_candidate={}
trains=[t[1:-1] for t in train['s2']]
import pickle
import OpenHowNet

hownet_dict = OpenHowNet.HowNetDict()


#trains=dataset.train_seqs
#dict=dataset.dict
#inv_dict=dataset.inv_dict


f=open('sss_dict.pkl','rb')
NNS,NNPS,JJR,JJS,RBR,RBS,VBD,VBG,VBN,VBP,VBZ,inv_NNS,inv_NNPS,inv_JJR,inv_JJS,inv_RBR,inv_RBS,inv_VBD,inv_VBG,inv_VBN,inv_VBP,inv_VBZ=pickle.load(f)
pos_list = ['noun', 'verb', 'adj', 'adv']
pos_set = set(pos_list)
'''
def count(s1,s2):
    not_in_num = 0
    for sem in s2:
        if sem not in s1:
            hy_flag = 0
            for sem2 in s1:
                if sem2 in hy_dict:
                    if sem in hy_dict[sem2]:
                        hy_flag = 1
                        break
            if hy_flag == 1:
                continue
            else:
                not_in_num += 1
                break
    return not_in_num
'''
s_ls = ['NNS', 'NNPS', 'JJR', 'JJS', 'RBR', 'RBS', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
s_noun=['NNS','NNPS']
s_verb=['VBD','VBG','VBN','VBP','VBZ']
s_adj=['JJR','JJS']
s_adv=['RBR','RBS']
def add_w1(w1,i1):
    word_candidate[i1] = {}
    w1_s_flag=0
    w1_orig=None
    for s in s_ls:
        if w1 in eval(s):
            w1_s_flag=1
            w1_orig = eval(s)[w1]
            break
    if w1_s_flag==0:
        w1_orig=w1
    result_list = hownet_dict.get(w1_orig)
    w1_pos = set()
    for pos in pos_set:
        word_candidate[i1][pos] = []
    for a in result_list:
        if type(a)!=dict:
            continue
        w1_pos.add(a['en_grammar'])
    #print(w1_pos)
    if len(w1_pos & pos_set) == 0:
        return
    if w1_orig=='CPR' or w1_orig=='*':
        return
    try:
        w1_sememes = hownet_dict.get_sememes_by_word(w1_orig, structured=False, lang="en", merge=False)
    except:
        w1_sememes = []
    new_w1_sememes=[t['sememes'] for t in w1_sememes ]
    #print(w1_sememes)
    if len(w1_sememes) == 0:
        return
    #print(new_w1_sememes)
   # for w2 in ['human', 'portrait', 'opera', 'actor', 'melodrama', 'role', 'soul', 'person', 'era', 'cartoon', 'farce', 'intensity', 'patience', 'pianist', 'image', 'creativity', 'morality', 'hallmark', 'trademark', 'attitude', 'parable', 'fable', 'comedian', 'category', 'chord', 'angel', 'scope', 'importance', 'rating', 'punk', 'circus', 'definition', 'objectivity', 'myth', 'spontaneity', 'artistry', 'information', 'continuity', 'spirituality', 'trilogy', 'ingenuity', 'inventiveness', 'randomness', 'craftsmanship', 'jazz', 'significance', 'entity', 'fantasia', 'storyteller', 'letter', 'candor', 'marker', 'corpus', 'newness', 'violinist', 'perceptiveness', 'propriety', 'humankind', 'magician', 'missive', 'stripe', 'epitaph', 'apocalypse', 'rendition', 'ditty', 'efficiency', 'playbook', 'spiritualism', 'aptitude', 'benchmark', 'mythology', 'intermezzo', 'burlesque', 'vacuum', 'coherence', 'ingredient']:
    for w2,i2 in vocab.items():   
        #i2=dataset.dict[w2]
        if i1 == i2:
            continue
        w2_s_flag = 0
        w2_orig = None
        w2_pos_sem=None
        for s in s_ls:
            if w2 in eval(s):
                w2_s_flag = 1
                w2_pos_sem=s
                w2_orig = eval(s)[w2]
                break
        if w2_s_flag == 0:
            w2_orig = w2
            w2_pos_sem='orig'
        result_list = hownet_dict.get(w2_orig)
        w2_pos = set()
        for a in result_list:
            if type(a)!=dict:
                continue
            w2_pos.add(a['en_grammar'])
        all_pos=w2_pos & w1_pos&pos_set
        if len(all_pos) == 0:
            continue
        #print(w2_orig)
        if w2_orig=='CPR' or w2_orig=='*':
            continue
        try:
            w2_sememes = hownet_dict.get_sememes_by_word(w2_orig, structured=False, lang="en", merge=False)
        except:
            w2_sememes=[]
        new_w2_sememes = [t['sememes'] for t in w2_sememes]
        #print(w2)
        #print(new_w1_sememes)
        #print(new_w2_sememes)
        if len(w2_sememes) == 0:
            continue
        #not_in_num1 = count(w1_sememes, w2_sememes)
        #not_in_num2 = count(w2_sememes,w1_sememes)
        #not_in_num=not_in_num1+not_in_num2
        can_be_sub=False
        for s1 in new_w1_sememes:
            for s2 in new_w2_sememes:
                
                if s1==s2:
                    can_be_sub=True
                    break      
        if can_be_sub == True:
            #print(w2,w2_pos,w2_sememes)
            for pos_valid in all_pos:
                s_flag = 0
                p_w1 = None
                for p in eval('s_' + pos_valid):
                    if w1 in eval(p):
                        s_flag = 1
                        p_w1 = p
                #print(s_flag,p_w1)
                if s_flag == 0:
                    if w2_pos_sem=='orig':
                        word_candidate[i1][pos_valid].append(i2)
                else:
                    if w2 in eval(p_w1):
                        word_candidate[i1][pos_valid].append(i2)
'''
for w1,i1 in vocab.items():
    print(i1)
    #if i1<100 and i1>80:
    add_w1(w1,i1)
'''
w='delicious'
add_w1(w,vocab[w])
#print(word_candidate[dataset.dict[w]])
print([inv_dict[t] for t in word_candidate[vocab[w]]['adj']])
w='good'
add_w1(w,vocab[w])
#print(word_candidate[dataset.dict[w]])
print([inv_dict[t] for t in word_candidate[vocab[w]]['adj']])
w='character'
add_w1(w,vocab[w])
#print(word_candidate[dataset.dict[w]])
print([inv_dict[t] for t in word_candidate[vocab[w]]['noun']])
'''
f=open('word_candidates_sense.pkl','wb')
pickle.dump(word_candidate,f)
'''
