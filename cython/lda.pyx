from libc.stdlib cimport rand, RAND_MAX
from libc.stdlib cimport malloc, free
import random

cdef struct token:
    long doc
    long word


cdef class LDA:

    cdef long doc_num, topic_num, word_num, token_num
    cdef double alpha, beta
    cdef double *p
    cdef long *word_count
    cdef long *doc_count
    cdef long *topic_count
    cdef long *z
    cdef token *token_list
    cdef assign
    
    def __cinit__(self, doc_num, topic_num, word_num, token_list):

        # initialize basic variables
        self.doc_num = doc_num
        self.topic_num = topic_num
        self.word_num = word_num
        
        # initialize token_list
        self.token_list = <token*>malloc(len(token_list) * sizeof(token))
        for i in range(len(token_list)):
            self.token_list[i].doc = token_list[i][0]
            self.token_list[i].word = token_list[i][1]
    
        self.token_num = len(token_list)

        # initialize hyper params
        self.alpha = 50.0 / topic_num
        self.beta = 0.1
    
        # malloc 
        self.word_count = <long*>malloc(topic_num * word_num * sizeof(long))
        self.doc_count = <long*>malloc(topic_num * doc_num * sizeof(long))
        self.topic_count = <long*>malloc(topic_num * sizeof(long))
        self.z = <long*>malloc(self.token_num * sizeof (long))
        self.p = <double*>malloc(self.topic_num * sizeof(double))

        # initialize
        for w in range(self.word_num):
            for k in range(self.topic_num):
                self.word_count[w * self.topic_num + k] = 0

        for d in range(self.doc_num):
            for k in range(self.topic_num):
                self.doc_count[d * self.topic_num + k] = 0

        for k in range(topic_num):
            self.topic_count[k] = 0

        for t in range(self.token_num):
            self.z[t] = 0


        # add token information
        for i in range(self.token_num):
            self.z[i] = rand() % self.topic_num 
            self.topic_count[self.z[i]] += 1
            self.doc_count[self.token_list[i].doc * self.topic_num + self.z[i]] += 1
            self.word_count[self.token_list[i].word * self.topic_num + self.z[i]] += 1


    def update(self):
        for i in range(self.token_num):
            self.resample(i)


    cdef resample(self, long i):
        
        # remove from current topic
        self.doc_count[self.token_list[i].doc * self.topic_num + self.z[i]] -= 1
        self.word_count[self.token_list[i].word * self.topic_num + self.z[i]] -= 1
        self.topic_count[self.z[i]] -= 1

        for k in range(self.topic_num):
            self.p[k] = (float(self.word_count[self.token_list[i].word * self.topic_num + k]) + self.beta) * \
                     (float(self.doc_count[self.token_list[i].doc * self.topic_num + k]) + self.alpha) / \
                     (float(self.topic_count[k]) + self.word_num * self.beta)
            if k != 0: self.p[k] += self.p[k-1]

        u = random.random() * self.p[self.topic_num - 1]
        for k in range(self.topic_num):
            if u < self.p[k]: 
                self.z[i] = k
                break

        self.doc_count[self.token_list[i].doc * self.topic_num + self.z[i]] += 1
        self.word_count[self.token_list[i].word * self.topic_num + self.z[i]] += 1
        self.topic_count[self.z[i]] += 1

    def get_phi(self):
        phi = [[0.0 for d in range(self.word_num)] for k in range(self.topic_num)]
        for k in range(self.topic_num):
            sum = 0.0
            for j in range(self.word_num):
                phi[k][j] = self.beta + float(self.word_count[j * self.topic_num + k])
                sum += phi[k][j]

            sinv = 1.0 / float(sum)
            for j in range(self.word_num):
                phi[k][j] *= sinv
        
        return phi



