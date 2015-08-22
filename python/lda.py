#! /usr/bin/env python

import random

class LDA:

    def __init__(self, doc_num, topic_num, word_num, token_list):
        self.doc_num = doc_num
        self.topic_num = topic_num
        self.word_num = word_num
        self.token_list = token_list

        self.alpha = 50.0 / topic_num
        self.beta = 0.1
    
        # initialize
        self.word_count = [[0 for k in range(self.topic_num)] for w in range (self.word_num)]
        self.doc_count = [[0 for k in range(self.topic_num)] for d in range(self.doc_num)]
        self.topic_count = [0 for k in range(self.topic_num)]
        self.z = [0 for i in range(len(self.token_list))]

        # add token information
        for i in range(len(self.token_list)):
            assign = random.randint(0, self.topic_num - 1)
            self.doc_count[token_list[i][0]][assign] += 1
            self.word_count[token_list[i][1]][assign] += 1
            self.topic_count[assign] += 1
            self.z[i] = assign


    def update(self):
        for i in range(len(self.token_list)):
            self.resample(i)


    def resample(self, i):
        assign = self.z[i]
        
        # remove from current topic
        self.doc_count[self.token_list[i][0]][assign] -= 1
        self.word_count[self.token_list[i][1]][assign] -= 1
        self.topic_count[assign] -= 1

        p = [0] * self.topic_num
        for k in range(self.topic_num):
            p[k] = float(self.word_count[self.token_list[i][1]][k] + self.beta) * \
                     float(self.doc_count[self.token_list[i][0]][k] + self.alpha) / \
                     float(self.topic_count[k] + self.word_num * self.beta)
            if k != 0: p[k] += p[k-1]

        u = random.random() * p[self.topic_num - 1]
        for k in range(self.topic_num):
            if u < p[k]: 
                assign = k
                break

        self.doc_count[self.token_list[i][0]][assign] += 1
        self.word_count[self.token_list[i][1]][assign] += 1
        self.topic_count[assign] += 1
        self.z[i] = assign


    def get_phi(self):
        phi = [[0.0 for d in range(self.word_num)] for k in range(self.topic_num)]
        for k in range(self.topic_num):
            sum = 0.0
            for j in range(self.word_num):
                phi[k][j] = self.beta + float(self.word_count[j][k])
                sum += phi[k][j]

            sinv = 1.0 / float(sum)
            for j in range(self.word_num):
                phi[k][j] *= sinv
        
        return phi


