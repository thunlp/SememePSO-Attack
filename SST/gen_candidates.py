import pickle
import OpenHowNet

word_candidate = {}

with open('aux_files/dataset.pkl', 'rb') as fh:
    dataset = pickle.load(fh)

hownet_dict = OpenHowNet.HowNetDict()

f = open('sss_dict.pkl', 'rb')
NNS, NNPS, JJR, JJS, RBR, RBS, VBD, VBG, VBN, VBP, VBZ, inv_NNS, inv_NNPS, inv_JJR, inv_JJS, inv_RBR, inv_RBS, inv_VBD, inv_VBG, inv_VBN, inv_VBP, inv_VBZ = pickle.load(
    f)
pos_list = ['noun', 'verb', 'adj', 'adv']
pos_set = set(pos_list)

s_ls = ['NNS', 'NNPS', 'JJR', 'JJS', 'RBR', 'RBS', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
s_noun = ['NNS', 'NNPS']
s_verb = ['VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
s_adj = ['JJR', 'JJS']
s_adv = ['RBR', 'RBS']
word_pos = {}
word_sem = {}
for w1, i1 in dataset.dict.items():
    w1_s_flag = 0
    w1_orig = None
    for s in s_ls:
        if w1 in eval(s):
            w1_s_flag = 1
            w1_orig = eval(s)[w1]
            break
    if w1_s_flag == 0:
        w1_orig = w1
    try:
        tree = hownet_dict.get_sememes_by_word(w1_orig, merge=False, structured=True, lang="en")
        w1_sememes = hownet_dict.get_sememes_by_word(w1_orig, structured=False, lang="en", merge=False)
        new_w1_sememes = [t['sememes'] for t in w1_sememes]
        # print(tree)

        w1_pos_list = [x['word']['en_grammar'] for x in tree]
        word_pos[i1] = w1_pos_list
        word_sem[i1] = new_w1_sememes
        main_sememe_list = hownet_dict.get_sememes_by_word(w1_orig, merge=False, structured=False, lang='en',
                                                           expanded_layer=2)
    except:
        word_pos[i1] = []
        word_sem[i1] = []
        main_sememe_list = []
    # assert len(w1_pos_list)==len(new_w1_sememes)
    # assert len(w1_pos_list)==len(main_sememe_list)


def add_w1(w1, i1):
    word_candidate[i1] = {}
    w1_s_flag = 0
    w1_orig = None
    w1_pos_sem = None

    for s in s_ls:
        if w1 in eval(s):
            w1_s_flag = 1
            w1_pos_sem = s
            w1_orig = eval(s)[w1]
            break
    if w1_s_flag == 0:
        w1_orig = w1
        w1_pos_sem = 'orig'

    w1_pos = set(word_pos[i1])
    for pos in pos_set:
        word_candidate[i1][pos] = set()
    valid_pos_w1 = w1_pos & pos_set

    if len(valid_pos_w1) == 0:
        return

    new_w1_sememes = word_sem[i1]
    if len(new_w1_sememes) == 0:
        return

    for w2, i2 in dataset.dict.items():

        if i1 == i2:
            continue
        w2_s_flag = 0
        w2_orig = None
        w2_pos_sem = None
        for s in s_ls:
            if w2 in eval(s):
                w2_s_flag = 1
                w2_pos_sem = s
                w2_orig = eval(s)[w2]
                break
        if w2_s_flag == 0:
            w2_orig = w2
            w2_pos_sem = 'orig'

        w2_pos = set(word_pos[i2])
        all_pos = w2_pos & w1_pos & pos_set
        if len(all_pos) == 0:
            continue

        new_w2_sememes = word_sem[i2]
        # print(w2)
        # print(new_w1_sememes)
        # print(new_w2_sememes)
        if len(new_w2_sememes) == 0:
            continue
        # not_in_num1 = count(w1_sememes, w2_sememes)
        # not_in_num2 = count(w2_sememes,w1_sememes)
        # not_in_num=not_in_num1+not_in_num2
        can_be_sub = False
        for s1_id in range(len(new_w1_sememes)):
            pos_w1 = word_pos[i1][s1_id]
            s1 = set(new_w1_sememes[s1_id])
            if pos_w1 not in pos_set:
                continue
            for s2_id in range(len(new_w2_sememes)):
                pos_w2 = word_pos[i2][s2_id]
                s2 = set(new_w2_sememes[s2_id])
                if pos_w1 == pos_w2 and s1 == s2:
                    if w1_pos_sem == 'orig':
                        if w2_pos_sem == 'orig':
                            word_candidate[i1][pos_w1].add(i2)
                    else:
                        for p in eval('s_' + pos_w1):
                            if w1 in eval(p):
                                if w2 in eval(p):
                                    word_candidate[i1][pos_w1].add(i2)

w='delicious'
add_w1(w,dataset.dict[w])
#print(word_candidate[dataset.dict[w]])
print([dataset.inv_dict[t] for t in word_candidate[dataset.dict[w]]['adj']])
w='good'
add_w1(w,dataset.dict[w])
#print(word_candidate[dataset.dict[w]])
print([dataset.inv_dict[t] for t in word_candidate[dataset.dict[w]]['adj']])
w='character'
add_w1(w,dataset.dict[w])
#print(word_candidate[dataset.dict[w]])
print([dataset.inv_dict[t] for t in word_candidate[dataset.dict[w]]['noun']])


for w1,i1 in dataset.dict.items():

    print(i1)

    add_w1(w1, i1)


f = open('word_candidates_sense.pkl', 'wb')
pickle.dump(word_candidate, f)



