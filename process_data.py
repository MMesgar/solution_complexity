#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 19:01:37 2017

@author: Mohsen Mesgar
For solution complexity

"""
import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
rng = np.random.RandomState(seed = 234567)
import time

#%%%
def split_main_corpus(corpus_path, output_dir):
    '''
    This function split datasets based on the solution_size of problems
    input:
        corpus_path: path to the main corpus provided by Annie
        output_dir: we write split files into a output_dir 
    output:
        None
    '''
    corpus_sol_size_2_3 = []
    corpus_sol_size_4_5 = []
    corpus_sol_size_gr_5 = []
    
    with open(corpus_path,'rb') as c:
        corpus = c.read()
    lines = corpus.split("\n")
    
    prob_ids = set()
    tmp_output = []
    tmp_prob_id = -1
    for line in lines:
        if line != '':
            columns = line.split("\t")
            prob_id = int(columns[0])
            prob_ids.add(prob_id)
            if prob_id != tmp_prob_id:
                n = len(tmp_output)
                if n==2 or n==3:
                    corpus_sol_size_2_3 += tmp_output                    
                elif n==4 or n==5:
                    corpus_sol_size_4_5 += tmp_output
                elif n>5:
                    corpus_sol_size_gr_5 += tmp_output

                tmp_output = [line]
                tmp_prob_id = prob_id
            else:
                tmp_output.append(line)
    # add the last problem
    n = len(tmp_output)
    if n==2 or n==3:
        corpus_sol_size_2_3 += tmp_output                    
    elif n==4 or n==5:
        corpus_sol_size_4_5 += tmp_output
    elif n>5:
        corpus_sol_size_gr_5 += tmp_output
    
    print "number of problems in corpus_all: %d"%len(prob_ids)
    print "number of lines with solution size 2-3: %d"%len(corpus_sol_size_2_3)
    print "number of lines with solution size 4-5: %d"%len(corpus_sol_size_4_5)
    print "number of lines with solution size gr-5: %d"%len(corpus_sol_size_gr_5)
    
    with open(output_dir[0]+'corpus.txt','wb') as new_c:
        output = '\n'.join(corpus_sol_size_2_3)
        new_c.write(output)
        
    with open(output_dir[1]+'corpus.txt','wb') as new_c:
        output = '\n'.join(corpus_sol_size_4_5)
        new_c.write(output)

    with open(output_dir[2]+'corpus.txt','wb') as new_c:
        output = '\n'.join(corpus_sol_size_gr_5)
        new_c.write(output)
    
    
#%%
def load_problem_solutions(corpus_path, clean_string = True):
    output = defaultdict(lambda:{})
    with open(corpus_path,'rb') as c:
        corpus = c.read()
    lines = corpus.split("\n")
    processed_corpus = []
    for line in lines:
        if line != '':
            columns = line.split("\t")
            prob_id = columns[0]
            solution_id = columns[2]# starts from 0 and higher is more complex
            if clean_string == True:
                problem_text = clean_str(columns[3])
                solution_text = clean_str(columns[8])# 4 is original solution; 8 is tokenized and normalized
            else:
                problem_text = columns[3]
                solution_text = columns[8]# 4 is original solution; 8 is tokenized and normalized

            processed_line = [int(prob_id),int(solution_id),problem_text,solution_text]
            processed_corpus.append(processed_line)  
     
        
    # extract all prob_id 
    ids = set([instance[0] for instance in processed_corpus])
    
    
    #extract all solutions associated with each prob_id
    for faq_id in ids:
        associated_instances = [instance for instance in processed_corpus 
                                if instance[0]==faq_id]
        problem = set([instance[2] for instance in associated_instances])
        if len(problem)>1:
            print "Error: inconsistency in the problem_text of faq_id=%s"%faq_id
            return None
            
        problem = list(problem)[0].strip().lower()
        
        # solutions are ordered from the easiest one to more complex one
        for i in range(len(associated_instances)):
            solution_txt = associated_instances[i][3].strip().lower()
            solution_id = associated_instances[i][1]
            output[faq_id]['prob_txt'] = problem
            output[faq_id][solution_id]=solution_txt
    return output
    
#%%
def process_orginal_corpus_to_triplets(input_path, output_path):
    with open(input_path,'rb') as c:
        corpus = c.read()
    lines = corpus.split("\n")
    processed_corpus = []
    prob_lens= []
    solution_lens = []
    for line in lines:
        if line != '':
            columns = line.split("\t")
            prob_id = columns[0]
            solution_id = columns[2]
            problem_text = columns[3]
            solution_text = columns[4]# 4 is original solution; 8 is tokenized and normalized
            processed_line = [int(prob_id), int(solution_id), problem_text, solution_text]
            processed_corpus.append(processed_line)
            prob_lens.append(len(problem_text.split()))
            solution_lens.append(len(solution_text.split()))
    print "number of lines in processed_corpus:%d"%len(processed_corpus)  
    print "average length of problems: %f"%np.mean(prob_lens)
    print "average length of solutions: %f"%np.mean(solution_lens)
    
    # extract all prob_id which is located in instance[0]
    ids = set([instance[0] for instance in processed_corpus])
    print "number of problems = %d"%len(ids)
    #extract all solutions associated with each prob_id
    # score of each solution
    instances = []
    for faq_id in ids:
        associated_instances = [instance for instance in processed_corpus 
                                if instance[0]==faq_id]
                                
        problem = set([instance[2] for instance in associated_instances])
        if len(problem)>1:
            print "Error: inconsistency in the problem_text of faq_id=%s"%faq_id
            return None
            
        problem = list(problem)[0]
        # solutions are ordered from the easiest one to more complex one
        for i in range(len(associated_instances)):
            easy_solution = associated_instances[i][3]
            easy_solution_id = associated_instances[i][1]
            for j in range(i + 1, len(associated_instances)):
                complex_solution = associated_instances[j][3]
                complex_solution_id = associated_instances[j][1]
                # build the triplet
                triplet = (1, faq_id, problem,  easy_solution_id, easy_solution, complex_solution_id, complex_solution)
                instances.append(triplet)
                triplet = (0, faq_id, problem, complex_solution_id, complex_solution, easy_solution_id, easy_solution)
                instances.append(triplet)
    #output elements of the triplet tab seprated in the a file
    output = []
    for instance in instances:
        line =''
        label = str(instance[0])
        faq_id= str(instance[1])
        problem_text = instance[2]
        solution_id1 = str(instance[3])
        solution_text1 = instance[4]
        solution_id2 = str(instance[5])
        solution_text2 = instance[6]
        line =   label + '\t' + faq_id + '\t'+ problem_text + '\t' + solution_id1 +'\t' + \
        solution_text1 + '\t' + solution_id2+ '\t' + solution_text2
        output.append(line)
    print "number of instances in dataset =%d"%len(output)
    
    with open(output_path,'wb') as dataset:
        output_text = '\n'.join(output)
        dataset.write(output_text)
#%%    
def process_original_corpus(input_path, output_path):
    with open(input_path,'rb') as c:
        corpus = c.read()
    lines = corpus.split("\n")
    processed_corpus = []
    for line in lines:
        if line != '':
            columns = line.split("\t")
            faq_id = columns[0]
            solution_id = columns[2]
            problem_text = columns[3]
            solution_text = columns[8]# 4 is original solution; 8 is tokenized and normalized
            processed_line = [int(faq_id),int(solution_id),problem_text,solution_text]
            processed_corpus.append(processed_line)
    print "number of lines in processed_corpus:%d"%len(processed_corpus)    
    # compute the complexity score and add it to each processed line
    #extract all faq_id
    ids = set([instance[0] for instance in processed_corpus])
    print "number of problems = %d"%len(ids)
    
    #extract all solutions associated to each faq_id and compute the complexity 
    # score of each solution
    instances = []
    for faq_id in ids:
        associated_solutions = [instance for instance in processed_corpus 
                                if instance[0]==faq_id]
        solution_ids  = [solution[1] for solution in associated_solutions]
        max_id = np.max(solution_ids)
        min_id = np.min(solution_ids)
        for associated_solution in associated_solutions:
            solution_id = associated_solution[1]
            solution_complexisty = (solution_id - 0.0) / float(max_id- min_id)
            associated_solution.append(solution_complexisty)
            instances.append(associated_solution)
    print "number of instances = %d"%len(instances)
    
    #output problem tab solution tab complexity_score in a file
    output = []
    for instance in instances:
        line =''
        problem_text = instance[2]
        solution_text = instance[3]
        complexity_score = instance[4]
        line =  str(complexity_score) + '\t' + problem_text + '\t' + solution_text
        output.append(line)
    print "number of instances in dataset =%d"%len(output)
    
    with open(output_path,'wb') as dataset:
        output_text = '\n'.join(output)
        dataset.write(output_text)
    
#%%
def build_triplet_datasets(data_folder, cv = 10, clean_string = True):
    """
    Loads data
    """
    revs = []
    dataset_file = data_folder[0]

    vocab = defaultdict(float)
    with open(dataset_file, "rb") as f:
        lines = f.readlines()
    num_lines = len(lines)
    
    ###
    #compute folds based on problems
    ###
    prob_ids = list(set([l.split('\t')[1] for l in lines]))
    num_prob = len(prob_ids)
    
    #print prob_ids
    #take 0.8 of problems for training and 0.1 for dev and 0.1 for test
    #num_train_prob= int(0.8 * num_prob)
    #num_dev_prob = int((num_prob - num_train_prob) / 2)
    #num_test_prob = num_prob - num_train_prob - num_dev_prob 

    #define folds
    permuted_prob_ids = rng.permutation(prob_ids)
    fold_size = int(num_prob / cv)
    folds = [permuted_prob_ids[k*fold_size:(k+1)*fold_size] for k in range(cv)]
    remaining= permuted_prob_ids[(cv)*fold_size:]
    cvs = []
    for i in range(len(folds)):
        test_probs = folds[i]
        j = (i+1) % len(folds) # choose next fold as validation set
        dev_probs = folds[j]
        train_probs =  []
        for k in range(len(folds)):
            if k!= i and k!=j:
                train_probs += list(folds[k])
        train_probs += list(remaining)
        #print len(train_probs) + len(dev_probs) + len(test_probs)
        cv_i = (train_probs, dev_probs, test_probs)
        cvs.append(cv_i)
    
    # now we process instances
    i = 0
    while (i<num_lines):
        line = lines[i]
        
        label = line.split('\t')[0]
        prob_id = line.split('\t')[1]
        sent1 = line.split('\t')[2]
        sent2_id = line.split('\t')[3]
        sent2 = line.split('\t')[4]
        sent3_id  = line.split('\t')[5]
        sent3 = line.split('\t')[6]
        rev1,rev2, rev3 = [],[], []
        rev1.append(sent1.strip())
        rev2.append(sent2.strip())
        rev3.append(sent3.strip())
        if clean_string:
            orig_rev1 = clean_str(" ".join(rev1))
            orig_rev2 = clean_str(" ".join(rev2))
            orig_rev3 = clean_str(" ".join(rev3))
        else:
            orig_rev1 = " ".join(rev1).lower()
            orig_rev2 = " ".join(rev2).lower()
            orig_rev3 = " ".join(rev3).lower()
        words1 = set(orig_rev1.split())
        words2 = set(orig_rev2.split())
        words3 = set(orig_rev3.split())
        for word in words1:
                vocab[word] += 1
        for word in words2:
                vocab[word] += 1
        for word in words3:
            vocab[word] += 1
        datum  = {"y":label, 
                  "pid":prob_id,
                  "text1": orig_rev1,
                  "id2": sent2_id,
                  "text2": orig_rev2,
                  "id3": sent3_id,
                  "text3": orig_rev3,
                  "num_words": np.max([len(orig_rev1.split()),len(orig_rev2.split()),len(orig_rev3.split())]),
                  "split": rng.randint(0,cv)
                     }    
        #print datum
        revs.append(datum)
        i += 1
    dataset = revs
    return dataset, vocab, cvs
    
#%%
def build_sent_pair_datasets(data_folder, cv = 10, clean_string = True):
    """
    Loads data
    """
    revs = []
    dataset_file = data_folder[0]

    vocab = defaultdict(float)
    with open(dataset_file, "rb") as f:
        lines = f.readlines()
    num_lines = len(lines)
    
    i = 0
    while (i<num_lines):
        line = lines[i]
        
        label = line.split('\t')[0]
        sent1 = line.split('\t')[1]
        sent2 = line.split('\t')[2]
        rev1,rev2 = [],[]
        rev1.append(sent1.strip())
        rev2.append(sent2.strip())
        if clean_string:
            orig_rev1 = clean_str(" ".join(rev1))
            orig_rev2 = clean_str(" ".join(rev2))
        else:
            orig_rev1 = " ".join(rev1).lower()
            orig_rev2 = " ".join(rev2).lower()
        words1 = set(orig_rev1.split())
        words2 = set(orig_rev2.split())
        for word in words1:
                vocab[word] += 1
        for word in words2:
                vocab[word] += 1
        datum  = {"y":label, 
                  "text1": orig_rev1,
                  "text2": orig_rev2, 
                  "num_words": np.max([len(orig_rev1.split()),len(orig_rev2.split())]),
                  "split": rng.randint(0,cv)
                     }    
        #print datum
        revs.append(datum)
        i += 1
    dataset = revs
    return dataset, vocab
    
#%%
def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
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
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs
#%%
def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  
#%%
def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
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
#%%   
def main_solution_score():
        # intitalize variables
    corpus_path = "./data/corpus.txt"
    dataset_path = "./data/dataset.txt"
    w2v_file = './data/GoogleNews-vectors-negative300.bin'
    data_folder = [dataset_path]
    dataset_pair_path = "./data/dataset_pairs.p"
    
    # process the original corpus to and compute the complexity score of each solution
    process_original_corpus(corpus_path, dataset_path)
    
    # do cross validation over instances because  it is possible that for a given problem we get 
    #a new solution and we would like to rank it
    # and pad it and extract all vocabularies          
    print "loading data...",        
    datasets, vocab = build_sent_pair_datasets(data_folder,cv=10, clean_string=True)
    max_num_words = np.max(pd.DataFrame(datasets)["num_words"])
    print "data loaded!"
    print "number of instances in dataset: " + str(len(datasets))
    print "vocab size: " + str(len(vocab))
    print "max sentence length (max_l): " + str(max_num_words)
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec is loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
 
    cPickle.dump([datasets, W, W2, word_idx_map, vocab], open(dataset_pair_path, "wb"))
    print "dataset_pairs created in %s!"%(dataset_pair_path)
#%%    
def main_solution_triplet(dir):
    # intitalize variables

    print "processing %s ..."%dir
    corpus_path = dir+"corpus.txt"
    dataset_path = dir+"triplet_dataset.txt"
    w2v_file = './data/GoogleNews-vectors-negative300.bin'
    data_folder = [dataset_path]
    dataset_triplet_path = dir+"dataset_triplet.p"
    
    # process the original corpus
    process_orginal_corpus_to_triplets(corpus_path, dataset_path)
    
    # do cross validation over instances 
    # and pad it and extract all vocabularies          
    print "loading data...",        
    datasets, vocab, folds = build_triplet_datasets(data_folder,cv=10, clean_string=True)
    max_num_words = np.max(pd.DataFrame(datasets)["num_words"])
    print "data loaded!"
    print "number of instances in dataset: " + str(len(datasets))
    print "vocab size: " + str(len(vocab))
    print "max sentence length (max_l): " + str(max_num_words)
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec is loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
 
    cPickle.dump([datasets, W, W2, word_idx_map, vocab, folds], open(dataset_triplet_path, "wb"))
    print "dataset_triplet created in %s!"%(dataset_triplet_path)
    
#%%
def main_corpus_split():
    split_main_corpus(corpus_path = "./data/corpus_all/corpus.txt", output_dir=['./data/corpus_2_3/','./data/corpus_4_5/','./data/corpus_gr_5/'])
#%%
if __name__=='__main__':
    print "local start time :", time.asctime(time.localtime(time.time()) )
    main_corpus_split()
    for dir in ['./data/corpus_all/', './data/corpus_2_3/','./data/corpus_4_5/','./data/corpus_gr_5/']:
    	main_solution_triplet(dir)
    print "local end time :", time.asctime(time.localtime(time.time()) )
    print('Done')
