import numpy as np
import pickle
from time import time

from attack_dpso_sem import PSOAttack

from torch.autograd import Variable
import torch
import torch.nn as nn

with open('./nli_tokenizer.pkl', 'rb') as fh:
    tokenizer = pickle.load(fh)

from model_nli import Model

with open('word_candidates_sense.pkl','rb') as fp:
    word_candidate=pickle.load(fp)

with open('./all_seqs.pkl', 'rb') as fh:
    train,valid,test = pickle.load(fh)
with open('pos_tags_test.pkl','rb') as fp:
    test_pos_tags=pickle.load(fp)
test_s1=[t[1:-1] for t in test['s1']]
test_s2=[t[1:-1] for t in test['s2']]
model=Model()
model.evaluate(test['s1'],test['s2'],test['label'])
np.random.seed(3333)
vocab = {w: i for (w, i) in tokenizer.word_index.items()}
inv_vocab = {i: w for (w, i) in vocab.items()}

#test_accuracy = model.evaluate([test[0], test[1]], test[2])[1]
#print('\nTest accuracy = ', test_accuracy)

adversary = PSOAttack(model, word_candidate,pop_size=60, max_iters=20)
print('the length of test cases is:', len(test_s1))
TEST_SIZE = 5000
test_idxs = np.random.choice(len(test_s1), size=TEST_SIZE, replace=False)
test_list = []
input_list = []
output_list = []
dist_list = []
test_times = []
success=[]
change_list=[]
target_list=[]
true_label_list=[]
success_count = 0
i = 0
while len(test_list) < 1000:
    print('\n')

    test_idx = test_idxs[i]
    s1=test_s1[test_idx]
    s2=test_s2[test_idx]
    pos_tags = test_pos_tags[test_idx]
    attack_pred = np.argmax(model.pred([s1],[s2])[0])
    true_label = test['label'][test_idx]
    x_len = np.sum(np.sign(s2))
    i += 1
    if attack_pred != true_label:
        print('Wrong classified')
    elif x_len<10:
        print('Skipping too short input')
    else:
        if true_label == 2:
            target = 0
        elif true_label == 0:
            target = 2
        else:
            target = 0 if np.random.uniform() < 0.5 else 2
        start_time = time()
        attack_result = adversary.attack(s1, s2, target,pos_tags)
        if attack_result is None:
            print('**** Attack failed **** ')
        else:
            num_changes = np.sum(np.array(s2) != np.array(attack_result))
            x_len = np.sum(np.sign(s2))

            print('%d - %d changed.' % (i + 1, int(num_changes)))
            modify_ratio = num_changes / x_len
            if modify_ratio > 0.25:
                print('too long:', modify_ratio)
            else:
                success_count += 1
                print('***** DONE ', len(test_list), '------')
                # test_times.append(time() - start_time)
                true_label_list.append(true_label)
                input_list.append([s1, s2, true_label])
                output_list.append(attack_result)
                success.append(test_idx)
                target_list.append(target)
                change_list.append(modify_ratio)
        test_list.append(test_idx)

print('Success rate: ', (success_count / len(test_list)))
f = open('AD_dpso_sem.pkl', 'wb')
pickle.dump((true_label_list, output_list, success, change_list, target_list), f)