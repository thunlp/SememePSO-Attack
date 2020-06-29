import pickle
import OpenHowNet

word_candidate = {}

with open('./nli_tokenizer.pkl', 'rb') as fh:
    tokenizer = pickle.load(fh)
vocab = {w: i for (w, i) in tokenizer.word_index.items()}
inv_vocab = {i: w for (w, i) in vocab.items()}

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
word_influction={}
for w1, i1 in vocab.items():
    w1_s_flag = 0
    w1_orig = None
    for s in s_ls:
        if w1 in eval(s):
            w1_s_flag = 1
            w1_orig = eval(s)[w1]
            word_influction[i1]=s
            break
    if w1_s_flag == 0:
        w1_orig = w1
        word_influction[i1] = 'orig'
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
    w1_pos_sem=word_influction[i1]



    w1_pos = set(word_pos[i1])
    for pos in pos_set:
        word_candidate[i1][pos] = []
    valid_pos_w1 = w1_pos & pos_set

    if len(valid_pos_w1) == 0:
        return

    new_w1_sememes = word_sem[i1]
    if len(new_w1_sememes) == 0:
        return

    for w2, i2 in vocab.items():
        if i2>50000:
            break
        if i1 == i2:
            continue
        w2_pos_sem=word_influction[i2]

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
        w_flag=0

        for s1_id in range(len(new_w1_sememes)):
            if w_flag == 1:
                break
            pos_w1 = word_pos[i1][s1_id]
            s1 = set(new_w1_sememes[s1_id])
            if pos_w1 not in pos_set:
                continue
            for s2_id in range(len(new_w2_sememes)):
                if w_flag==1:
                    break
                pos_w2 = word_pos[i2][s2_id]
                s2 = set(new_w2_sememes[s2_id])
                if pos_w1 == pos_w2 and s1 == s2:
                    if w1_pos_sem == 'orig':
                        if w2_pos_sem == 'orig':
                            word_candidate[i1][pos_w1].append(i2)
                            w_flag=1
                            break
                    else:
                        for p in eval('s_' + pos_w1):
                            if w1 in eval(p) and w2 in eval(p):
                                word_candidate[i1][pos_w1].append(i2)
                                w_flag=1
                                break




for w1,i1 in vocab.items():
    if i1>50000:
        break
    print(i1)

    add_w1(w1, i1)



f = open('word_candidates_sense.pkl', 'wb')
pickle.dump(word_candidate, f)



