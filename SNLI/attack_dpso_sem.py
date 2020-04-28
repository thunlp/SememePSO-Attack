from __future__ import division
import numpy as np

import tensorflow as tf
import copy


class PSOAttack(object):
    def __init__(self, model, candidate,
                 pop_size=20, max_iters=100):
        self.candidate = candidate

        self.model = model
        self.max_iters = max_iters
        self.pop_size = pop_size

        self.temp = 0.3

    def do_replace(self, x_cur, pos, new_word):
        x_new = x_cur.copy()
        x_new[pos] = new_word
        return x_new

    def mutate(self, x_cur, w_select_probs, w_list):
        x_len = w_select_probs.shape[0]
        # print('w_select_probs:',w_select_probs)
        rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        return self.do_replace(x_cur, rand_idx, w_list[rand_idx])

    def generate_population(self, x1,x_orig, neigbhours_list, target, pop_size, x_len, neighbours_len):
        h_score, w_list = self.gen_h_score(x1,x_len, target, neighbours_len, neigbhours_list, x_orig)
        return [self.mutate(x_orig, h_score, w_list) for _ in
                range(pop_size)]

    def turn(self, x1, x2, prob, x_len):
        x_new = copy.deepcopy(x2)
        for i in range(x_len):
            if np.random.uniform() < prob[i]:
                x_new[i] = x1[i]
        return x_new

    def gen_most_change(self, x1,pos, x_cur, target, replace_list):

        new_x_list = [self.do_replace(x_cur, pos, w) if x_cur[pos] != w and w != 0 else x_cur for w in replace_list]
        pop_x1 = self.self_copy(x1, len(new_x_list))
        new_x_preds = self.model.pred(pop_x1,new_x_list)
        new_x_scores = new_x_preds[:, target]
        orig_score = self.model.pred([x1],[x_cur])[0, target]
        new_x_scores = new_x_scores - orig_score
        return np.max(new_x_scores), new_x_list[np.argsort(new_x_scores)[-1]][pos]

    def softmax(self, n):

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

    def gen_h_score(self, x1,x_len, target, neighbours_len, neigbhours_list, x_now):

        w_list = []
        prob_list = []
        for i in range(x_len):
            if neighbours_len[i] == 0:
                w_list.append(x_now[i])
                prob_list.append(0)
                continue
            p, w = self.gen_most_change(x1,i, x_now, target, neigbhours_list[i])
            w_list.append(w)
            prob_list.append(p)

        prob_list = self.softmax(prob_list)
        # print('neighbours_len:',neighbours_len)
        # print('prob_list:',prob_list)

        h_score = prob_list
        h_score = np.array(h_score)
        return h_score, w_list

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

    def cal_fitness(self, pop, x_orig, x_len, orig_score, pop_scores):
        change_ratio = [self.count_change_ratio(p, x_orig, x_len) for p in pop]
        for t in range(len(change_ratio)):
            if change_ratio[t] == 0:
                change_ratio[t] = float(1 / x_len)
        preds_change = [p - orig_score for p in pop_scores]
        fitness = []
        for i in range(len(pop)):
            fitness.append(preds_change[i] / change_ratio[i])
        return fitness

    def cal_u(self, S1, S2, alpha, rank):
        if rank <= S1 * self.pop_size:
            u1 = 1 + alpha
            u2 = 1 - alpha
        elif rank > S1 * self.pop_size and rank < S2 * self.pop_size:
            u1 = 1
            u2 = 1
        else:
            u1 = 1 - alpha
            u2 = 1 + alpha
        return u1, u2
    def self_copy(self,x,n):
        new_x=[x for i in range(n)]
        return new_x
    def attack(self, x1,x_orig, target, pos_tags):
        x_adv = x_orig.copy()
        pop_x1=self.self_copy(x1,self.pop_size)
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

        if np.sum(neighbours_len) == 0:
            return None

        print(neighbours_len)

        pop = self.generate_population(x1,x_orig, neigbhours_list, target, self.pop_size, x_len, neighbours_len)
        part_elites = copy.deepcopy(pop)

        pop_preds = self.model.pred(pop_x1,pop)
        pop_scores = pop_preds[:, target]
        part_elites_scores = pop_scores
        all_elite_score = np.max(pop_scores)
        pop_ranks = np.argsort(pop_scores)[::-1]
        top_attack = pop_ranks[0]
        all_elite = pop[top_attack]
        if np.argmax(pop_preds[top_attack, :]) == target:
            return pop[top_attack]
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

            '''
            pop_fitness=self.cal_fitness(pop,x_orig,x_len,orig_score,pop_scores)

            fitness_rank=list(np.argsort(pop_fitness))
            fitness_rank.reverse()

            id_rank={}
            for rr in range(len(fitness_rank)):
                id_rank[fitness_rank[rr]]=rr
            '''
            for id in range(self.pop_size):
                # U1,U2=self.cal_u(S1,S2,alpha,id_rank[id])
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
                # pop[id]=self.turn(part_elites[id],pop[id],turn_prob,x_len)
                # pop[id]=self.turn(all_elite,pop[id],turn_prob,x_len)
            pop_preds = self.model.pred(pop_x1,pop)
            pop_scores = pop_preds[:, target]
            pop_ranks = np.argsort(pop_scores)[::-1]
            top_attack = pop_ranks[0]

            print('\t\t', i, ' -- ', 'before mutation', pop_scores[top_attack])
            if np.argmax(pop_preds[top_attack, :]) == target:
                return pop[top_attack]

            new_pop = []
            for x in pop:
                change_ratio = self.count_change_ratio(x, x_orig, x_len)
                p_change = 1 - 2*change_ratio
                if np.random.uniform() < p_change:
                    new_h, new_w_list = self.gen_h_score(x1,x_len, target, neighbours_len, neigbhours_list, x)
                    new_pop.append(self.mutate(x, new_h, new_w_list))
                else:
                    new_pop.append(x)
            pop = new_pop

            pop_preds = self.model.pred(pop_x1,pop)
            pop_scores = pop_preds[:, target]
            pop_ranks = np.argsort(pop_scores)[::-1]
            top_attack = pop_ranks[0]

            print('\t\t', i, ' -- ', 'after mutation', pop_scores[top_attack])
            if np.argmax(pop_preds[top_attack, :]) == target:
                return pop[top_attack]
            for k in range(self.pop_size):
                if pop_scores[k] > part_elites_scores[k]:
                    part_elites[k] = pop[k]
                    part_elites_scores[k] = pop_scores[k]
            elite = pop[top_attack]
            if np.max(pop_scores) > all_elite_score:
                all_elite = elite
                all_elite_score = np.max(pop_scores)
        return None
