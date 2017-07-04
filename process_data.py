# -*- coding: utf-8 -*-
import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd

vector_size = 50
def build_data_cv(data_folder, cv=5, clean_string=True):
    """
    二分类数据预处理
    Loads data and split into 5 folds.
    """
    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    third_file=data_folder[2]

    vocab = defaultdict(float)
    count = 0
    with open(pos_file, "rb") as f:
        for line in f:
            if len(line)<15:
                continue
            rev = []
            rev.append(line.strip())
            #print rev
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            #lable [0,1,2..]的一个值
            datum  = {"y":1,#lable
                      "text": orig_rev,#原始文本                             
                      "num_words": len(orig_rev.split()),#该段文本的单词数量
                      "split": np.random.randint(0,cv)}#具体的CV值
            revs.append(datum)
            count = count + 1
            if count>200000:
                break
    count = 0
    with open(neg_file, "rb") as f:
        for line in f:       
            if len(line)<15:
                continue
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":0, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
            count = count + 1
            if count>200000:
                break
    count = 0
    with open(third_file, "rb") as f:
        for line in f:
            if len(line) < 15:
                continue
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": 2,
                     "text": orig_rev,
                     "num_words": len(orig_rev.split()),
                     "split": np.random.randint(0, cv)}
            revs.append(datum)
            count = count + 1
            if count > 200000:
                break

    return revs, vocab #返回原始的文本及对其的标记、单词表{word : word_count}

def build_data_cv_multi(data_folder, cv=5, clean_string=False):
    """
    多分类数据预处理
    Loads data and split into 5 folds.
    """
    revs = []
    c08 = data_folder[0]
    vocab = defaultdict(float)
    with open(c08, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            #lable [0,1,2..]的一个值
            datum  = {"y":0,
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
            
    c10 = data_folder[1]
    with open(c10, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":1, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)

    c13 = data_folder[2]
    with open(c13, "rb") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":2,
                      "text": orig_rev,
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)

    c14 = data_folder[3]
    with open(c14, "rb") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": 3,
                     "text": orig_rev,
                     "num_words": len(orig_rev.split()),
                     "split": np.random.randint(0, cv)}
            revs.append(datum)

    c16 = data_folder[4]
    with open(c16, "rb") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": 4,
                     "text": orig_rev,
                     "num_words": len(orig_rev.split()),
                     "split": np.random.randint(0, cv)}
            revs.append(datum)

    # c20 = data_folder[5]
    # with open(c20, "rb") as f:
    #     for line in f:
    #         rev = []
    #         rev.append(line.strip())
    #         if clean_string:
    #             orig_rev = clean_str(" ".join(rev))
    #         else:
    #             orig_rev = " ".join(rev).lower()
    #         words = set(orig_rev.split())
    #         for word in words:
    #             vocab[word] += 1
    #         datum = {"y": 5,
    #                  "text": orig_rev,
    #                  "num_words": len(orig_rev.split()),
    #                  "split": np.random.randint(0, cv)}
    #         revs.append(datum)
    #
    # c22 = data_folder[6]
    # with open(c22, "rb") as f:
    #     for line in f:
    #         rev = []
    #         rev.append(line.strip())
    #         if clean_string:
    #             orig_rev = clean_str(" ".join(rev))
    #         else:
    #             orig_rev = " ".join(rev).lower()
    #         words = set(orig_rev.split())
    #         for word in words:
    #             vocab[word] += 1
    #         datum = {"y": 6,
    #                  "text": orig_rev,
    #                  "num_words": len(orig_rev.split()),
    #                  "split": np.random.randint(0, cv)}
    #         revs.append(datum)
    #
    # c23 = data_folder[7]
    # with open(c23, "rb") as f:
    #     for line in f:
    #         rev = []
    #         rev.append(line.strip())
    #         if clean_string:
    #             orig_rev = clean_str(" ".join(rev))
    #         else:
    #             orig_rev = " ".join(rev).lower()
    #         words = set(orig_rev.split())
    #         for word in words:
    #             vocab[word] += 1
    #         datum = {"y": 7,
    #                  "text": orig_rev,
    #                  "num_words": len(orig_rev.split()),
    #                  "split": np.random.randint(0, cv)}
    #         revs.append(datum)
    #
    # c24 = data_folder[8]
    # with open(c24, "rb") as f:
    #     for line in f:
    #         rev = []
    #         rev.append(line.strip())
    #         if clean_string:
    #             orig_rev = clean_str(" ".join(rev))
    #         else:
    #             orig_rev = " ".join(rev).lower()
    #         words = set(orig_rev.split())
    #         for word in words:
    #             vocab[word] += 1
    #         datum = {"y": 8,
    #                  "text": orig_rev,
    #                  "num_words": len(orig_rev.split()),
    #                  "split": np.random.randint(0, cv)}
    #         revs.append(datum)
    return revs, vocab
    
def get_W(word_vecs, k=vector_size):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float64')            
    W[0] = np.zeros(k, dtype='float64')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map #W为一个词向量矩阵 一个word可以通过word_idx_map得到其在W中的索引，进而得到其词向量

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    ''''
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float64').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float64')  
            else:
                f.read(binary_len)
    '''
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=vector_size):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    #string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

if __name__=="__main__":    
    w2v_file = ""     
    #data_folder = ["rt-polarity.pos","rt-polarity.neg"] 
    # data_folder = ["SogouC.reduced/Reduced/C000013_pre.txt",
    # "SogouC.reduced/Reduced/C000024_pre.txt"]
    #data_folder = ["SogouC.reduced/Reduced/lixia-test-1.txt","SogouC.reduced/Reduced/lixia-test-2.txt"]
    data_folder = [
                   "question/HUM.txt",
                   "question/LOC.txt",
                   "question/NUM.txt",
                   # "question/OBJ.txt",
                   "question/UNKNOW.txt" ,
                   "question/TIME.txt"
                   ]
    print "loading data...",
    revs, vocab = build_data_cv_multi(data_folder, cv=5, clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)#利用一个构建好的word2vec向量来初始化词向量矩阵及词-向量映射表
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab) #得到一个{word:word_vec}词典
    W2, _ = get_W(rand_vecs)#构建一个随机初始化的W2词向量矩阵
    cPickle.dump([revs, W, W2, word_idx_map, vocab], open("mr.p", "wb"))
    print "dataset created!"
    
