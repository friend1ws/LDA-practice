#! /usr/bin/evn python

import argparse
import lda

def read_bow_file(bow_file_path):
    
    hIN = open(bow_file_path, 'r')
    D = int(hIN.readline())
    W = int(hIN.readline())
    N = int(hIN.readline())

    token = []  
    for line in hIN:
        if line.startswith('#'): continue
        doc_id, word_id, count = line.rstrip('\n').split(' ')
        for i in range(int(count)):
            token.append((int(doc_id) - 1, int(word_id) - 1))

    return (token, D, W, N)


def read_vocab_file(vocab_data_path):

    hIN = open(vocab_data_path, 'r')

    word= []
    for line in hIN:
        if line.startswith('#'): continue
        word.append(line.rstrip('\n'))

    return word


def print_word_topic(phi, K, W, word):

    for k in range(K):

        print "topic: " + str(k)

        word2prob = {}
        for w in range(W):
            word2prob[word[w]] = phi[k][w]
   
    
        num = 0
        for k, v in sorted(word2prob.items(), key=lambda x:x[1], reverse = True):
            print k + '\t' + str(v)
            num += 1
            if num > 10: break

        print ""

 
def main(args):
     
    bow_file = args.bow_file
    vocab_file = args.vocab_file
    K = args.K
    iter_num = args.iter_num

    # read bow file
    token, D, W, N = read_bow_file(bow_file)

    # read vocabrary file
    word = read_vocab_file(vocab_file)

    # initialize LDA class
    learn_lda = lda.LDA(D, K, W, token)

    # Gibbs sampling
    for i in range(iter_num):
        learn_lda.update()
        if i % 1 == 0:
            print_word_topic(learn_lda.get_phi(), K, W, word)



      
