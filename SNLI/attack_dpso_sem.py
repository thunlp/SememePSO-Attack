from __future__ import division
import numpy as np

import tensorflow as tf
import copy


class PSOAttack(object):
    def __init__(self, model, candidate,
                 pop_size=60, max_iters=20):
        self.candidate = candidate
        self.invoke_dict={}
        self.model = model
        self.max_iters = max_iters
        self.pop_size = pop_size

        self.temp = 0.3


    def do_replace(self, x_cur, pos, new_word):
        x_new = x_cur.copy()
        x_new[pos] = new_word
        return x_new
    def predict_batch(self,x1,sentences):

        return np.array([self.predict(x1,s) for s in sentences])
    def predict(self,x1,sentence):
        if tuple(sentence) in self.invoke_dict:
            return self.invoke_dict[tuple(sentence)]
        tem=self.model.predict(np.array([x1]),np.array([sentence]))[0]
        self.invoke_dict[tuple(sentence)]=tem

        return tem
    def select_best_replacement(self, x1, pos, x_cur, x_orig, target, replace_list):
        """ Select the most effective replacement to word at pos (pos)
        in (x_cur) between the words in replace_list """
        new_x_list = [self.do_replace(
            x_cur, pos, w) if x_orig[pos] != w and w != 0 else x_cur for w in replace_list]
        new_x_preds = self.predict_batch(x1, new_x_list)
        # Keep only top_n
        # replace_list = replace_list[:self.top_n]
        # new_x_list = new_x_list[:self.top_n]
        # new_x_preds = new_x_preds[:self.top_n,:]
        x_scores = new_x_preds[:, target]
        orig_score = self.predict(x1,x_cur)[target]


        new_x_scores = x_scores - orig_score
        # Eliminate not that clsoe words

        if (np.max(new_x_scores) > 0):
            best_id = np.argsort(new_x_scores)[-1]
            if np.argmax(new_x_preds[best_id]) == target:
                return [1, new_x_list[best_id]]
            return [x_scores[best_id], new_x_list[best_id]]
        return [orig_score, x_cur]

    def perturb(self, x_cur, x_orig, neigbhours, w_select_probs, target):
        # Pick a word that is not modified and is not UNK
        x_len = w_select_probs.shape[0]
        # to_modify = [idx  for idx in range(x_len) if (x_cur[idx] == x_orig[idx] and self.inv_dict[x_cur[idx]] != 'UNK' and
        #                                             self.dist_mat[x_cur[idx]][x_cur[idx]] != 100000) and
        #                     x_cur[idx] not in self.skip_list
        #            ]
        rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        while x_cur[rand_idx] != x_orig[rand_idx] and np.sum(x_orig != x_cur) < np.sum(np.sign(w_select_probs)):

            rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]

        # src_word = x_cur[rand_idx]
        # replace_list,_ =  glove_utils.pick_most_similar_words(src_word, self.dist_mat, self.top_n, 0.5)
        replace_list = neigbhours[rand_idx]
        return self.select_best_replacement(rand_idx, x_cur, x_orig, target, replace_list)

    def generate_population(self, x_orig, neigbhours_list, w_select_probs, target, pop_size):
        pop = []
        pop_scores=[]
        for i in range(pop_size):
            tem = self.perturb(x_orig, x_orig, neigbhours_list, w_select_probs, target)
            if tem is None:
                return None
            if tem[0] == 1:
                return [tem[1]]
            else:
                pop_scores.append(tem[0])
                pop.append(tem[1])
        return pop_scores,pop

    def turn(self, x1, x2, prob, x_len):
        x_new = copy.deepcopy(x2)
        for i in range(x_len):
            if np.random.uniform() < prob[i]:
                x_new[i] = x1[i]
        return x_new


    def norm(self, n):

        tn = []
        for i in n:
            if i <= 0:
                tn.append(0)
            else:
                tn.append(i)
        s = np.sum(tn)
        if s == 0:
            for i in range(len(tn)):
                tn[i] = 1
            return [t / len(tn) for t in tn]
        new_n = [t / s for t in tn]

        return new_n



    def equal(self, a, b):
        if a == b:
            return -3
        else:
            return 3

    def sigmod(self, n):
        return 1 / (1 + np.exp(-n))

    def count_change_ratio(self, x, x_orig, x_len):
        change_ratio = float(np.sum(x != x_orig)) / float(x_len)
        return change_ratio
    '''
    def gen_pos_saliency(self,pos,x_orig,target,orig_score):
        text_orig=[self.inv_dict[t] for t in x_orig if t!=0]
        text_new=' '.join(self.do_replace(text_orig,pos,'<unk>'))
        s_new=self.predict_text(text_new)[target]

        saliency=orig_score-s_new
        return saliency
    '''
    def self_copy(self,x,n):
        new_x=[x for i in range(n)]
        return new_x
    def attack(self, x1, x_orig, target, pos_tags):
        pop_x1 = self.self_copy(x1, self.pop_size)
        self.invoke_dict = {}
        x_adv = x_orig.copy()
        x_len = np.sum(np.sign(x_orig))
        x_len = int(x_len)
        pos_list = ['JJ', 'NN', 'RB', 'VB']

        neigbhours_list = []
        for i in range(x_len):
            if x_adv[i] not in range(1, 50000):
                neigbhours_list.append([])
                continue
            pair = pos_tags[i]
            if pair[1][:2] not in pos_list:
                neigbhours_list.append([])
                continue
            if pair[1][:2] == 'JJ':
                pos = 'adj'
            elif pair[1][:2] == 'NN':
                pos = 'noun'
            elif pair[1][:2] == 'RB':
                pos = 'adv'
            else:
                pos = 'verb'
            if pos in self.candidate[x_adv[i]]:
                neigbhours_list.append([neighbor for neighbor in self.candidate[x_adv[i]][pos]])
            else:
                neigbhours_list.append([])

        neighbours_len = [len(x) for x in neigbhours_list]

        w_select_probs=[]
        for pos in range(x_len):
            if neighbours_len[pos]==0:
                w_select_probs.append(0)
            else:
                w_select_probs.append(min(neighbours_len[pos],10))
        w_select_probs=w_select_probs/np.sum(w_select_probs)

        orig_score=self.predict(x1,x_orig)
        print('orig',orig_score[target])

        if np.sum(neighbours_len) == 0:
            return None

        print(neighbours_len)



        tem = self.generate_population(x_orig, neigbhours_list, w_select_probs, target, self.pop_size)
        if tem is None:
            return None
        if len(tem)==1:
            return tem[0]
        pop_scores,pop=tem
        part_elites = copy.deepcopy(pop)
        part_elites_scores = pop_scores
        all_elite_score = np.max(pop_scores)
        pop_ranks = np.argsort(pop_scores)
        top_attack = pop_ranks[-1]
        all_elite = pop[top_attack]


        Omega_1 = 0.8
        Omega_2 = 0.2
        C1_origin = 0.8
        C2_origin = 0.2
        V = [np.random.uniform(-3, 3) for rrr in range(self.pop_size)]
        V_P = [[V[t] for rrr in range(x_len)] for t in range(self.pop_size)]

        for i in range(self.max_iters):

            Omega = (Omega_1 - Omega_2) * (self.max_iters - i) / self.max_iters + Omega_2
            C1 = C1_origin - i / self.max_iters * (C1_origin - C2_origin)
            C2 = C2_origin + i / self.max_iters * (C1_origin - C2_origin)


            for id in range(self.pop_size):

                for dim in range(x_len):
                    V_P[id][dim] = Omega * V_P[id][dim] + (1 - Omega) * (
                                self.equal(pop[id][dim], part_elites[id][dim]) + self.equal(pop[id][dim],
                                                                                            all_elite[dim]))
                turn_prob = [self.sigmod(V_P[id][d]) for d in range(x_len)]
                P1 = C1
                P2 = C2
                # P1=self.sigmod(P1)
                # P2=self.sigmod(P2)

                if np.random.uniform() < P1:
                    pop[id] = self.turn(part_elites[id], pop[id], turn_prob, x_len)
                if np.random.uniform() < P2:
                    pop[id] = self.turn(all_elite, pop[id], turn_prob, x_len)

            pop_scores = []
            pop_scores_all=[]
            for a in pop:
                pt = self.predict(x1,a)

                pop_scores.append(pt[target])
                pop_scores_all.append(pt)
            pop_ranks = np.argsort(pop_scores)
            top_attack = pop_ranks[-1]

            print('\t\t', i, ' -- ', pop_scores[top_attack])
            for pt_id in range(len(pop_scores_all)):
                pt = pop_scores_all[pt_id]
                if np.argmax(pt) == target:

                    return pop[pt_id]

            new_pop = []
            new_pop_scores=[]
            for id in range(len(pop)):
                x=pop[id]
                change_ratio = self.count_change_ratio(x, x_orig, x_len)
                p_change = 1 - 2*change_ratio
                if np.random.uniform() < p_change:
                    tem = self.perturb(x, x_orig, neigbhours_list, w_select_probs, target)
                    if tem is None:
                        return None
                    if tem[0] == 1:

                        return tem[1]
                    else:
                        new_pop_scores.append(tem[0])
                        new_pop.append(tem[1])
                else:
                    new_pop_scores.append(pop_scores[id])
                    new_pop.append(x)
            pop = new_pop

            pop_scores = new_pop_scores
            pop_ranks = np.argsort(pop_scores)
            top_attack = pop_ranks[-1]
            for k in range(self.pop_size):
                if pop_scores[k] > part_elites_scores[k]:
                    part_elites[k] = pop[k]
                    part_elites_scores[k] = pop_scores[k]
            elite = pop[top_attack]
            if np.max(pop_scores) > all_elite_score:
                all_elite = elite
                all_elite_score = np.max(pop_scores)
        return None

