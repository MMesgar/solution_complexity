#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 17:39:19 2017

@author: mesgarmn
"""
#%%
import theano
from collections import defaultdict
import numpy as np
import theano.tensor as T
#%%
def debug_print(var, name, PRINT_VARS = True):
    if PRINT_VARS is False:
        return var
    return theano.printing.Print(name)(var)
#%%
def compute_similarity(v1,v2):
    d = v1 - v2
    d2 = d**2
    s = 1.+ d2.sum(axis=1,keepdims=True)
    output = 1. / s
    return output
#%%
def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

#%%
def get_idx_from_sent(sent, word_idx_map, max_l=51, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

#%%
def make_idx_data_cv(rng, dataset, word_idx_map, fold,  max_l=51,filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train_probs, dev_probs, test_probs = map(int,fold[0]), map(int,fold[1]), map(int,fold[2])

    train_x,train_y, dev_x, dev_y, test_x, test_y= [],[],[],[],[],[]
    test_dict = defaultdict(lambda:{})
    for rev in dataset:
        instance_x = []
        prob_id = int(rev["pid"])
        sent1 = get_idx_from_sent(rev["text1"], word_idx_map, max_l,filter_h) 
        sent2_id = int(rev["id2"])
        sent2 = get_idx_from_sent(rev["text2"], word_idx_map, max_l,filter_h)
        sent3_id = int(rev["id3"])
        sent3 = get_idx_from_sent(rev["text3"], word_idx_map, max_l,filter_h)
        label = rev["y"]
        
        if prob_id in list(train_probs):
            instance_x = sent1 + sent2 + sent3
            train_x.append(instance_x)
            train_y.append(label)
        elif prob_id in list(dev_probs):
            instance_x = sent1 + sent2 + sent3
            dev_x.append(instance_x)
            dev_y.append(label)
        elif prob_id in list(test_probs):
###########################
## lines between 411 to 414 adds all samples to test (old version)
###########################
#            test_x.append(instance_x)
#            test_y.append(label)
#            test_dict[prob_id]['prob_txt']=rev["text1"]
#            test_dict[prob_id][rev["text2"]]=sent2_id
#            test_dict[prob_id][rev["text3"]]=sent3_id
###########################
## following part is new. 
##It adds only labels 1 (problem, easy, complex) instances. 
## 
###########################
            if int(label) ==1 :
                if rng.uniform()>0.5:
                    instance_x = sent1 + sent2 + sent3
                    test_x.append(instance_x)
                    test_y.append(label)
                    test_dict[prob_id]['prob_txt']=rev["text1"]
                    test_dict[prob_id][rev["text2"]]=sent2_id
                    test_dict[prob_id][rev["text3"]]=sent3_id
                else:
                    instance_x = sent1 +  sent3 + sent2
                    label = 0
                    test_x.append(instance_x)
                    test_y.append(label)
                    test_dict[prob_id]['prob_txt']=rev["text1"]
                    test_dict[prob_id][rev["text2"]]=sent2_id
                    test_dict[prob_id][rev["text3"]]=sent3_id
        else:
            print('Problem id %d is not in any places (train,dev,test set) in the fold.'%prob_id)
            raise         
    train = (np.array(train_x,dtype="int"), np.array(train_y,dtype='int'))
    validation = (np.array(dev_x,dtype="int"), np.array(dev_y,dtype='int'))
    test = (np.array(test_x,dtype="int"), np.array(test_y,dtype='int'))
    return [train, validation, test], test_dict  

#%%
def print_errors_in_file(labels_prob, errors, testset, testset_dict, word_idx_map, save_to):
    output = []
    idx_word_map = {k:v for v,k in word_idx_map.items()}
    test_x  = testset[0]
    test_y = testset[1]
    found_problems = []
    for i, inst in enumerate(test_x):
       probs_0 = labels_prob[i,0]
       probs_1 = labels_prob[i,1]
       error = errors[i]
       gold = test_y[i]
       len_each_sent = len(inst) / 3
       prob = inst[:len_each_sent]
       inst1 = inst[len_each_sent:2*len_each_sent]
       inst2 = inst[2*len_each_sent:] 
       problem = [idx_word_map[word_idx] for word_idx in prob if word_idx>0]
       sent1 = [idx_word_map[word_idx] for word_idx in inst1 if word_idx>0]
       sent2 = [idx_word_map[word_idx] for word_idx in inst2 if word_idx>0]
       problem  = ' '.join(problem)
       sent1 = ' '.join(sent1)
       sent2 = ' '.join(sent2)
       
       # find pid,
       problem_id = -1
       for pid, values in testset_dict.items():
           if values.has_key(sent1) and values.has_key(sent2) and values['prob_txt']==problem and problem_id==-1:
               problem_id = pid
               sent1_id = values[sent1]
               sent2_id = values[sent2]
       #make sure that we faound pid
       assert problem_id != -1 
       if problem_id not in found_problems:
           found_problems.append(problem_id)
       
       sent = str(problem_id) +'@@' + problem + '@@' + \
       str(sent1_id)+'@@' + sent1 + '@@' +str(sent2_id) +'@@'+ sent2
       
       if error == 1:
           line = str(probs_0) +'@@' + str(probs_1) + '@@'+ str(1-gold) + '@@'+ str(gold) + '@@'+ sent
       else:
           line = str(probs_0) +'@@' + str(probs_1) + '@@'+ str(gold) + '@@'+ str(gold) + '@@'+ sent 
        
       output.append(line)
       
    content = '\n'.join(output)
    with open(save_to, 'w') as err_file:
       err_file.write(content)
    return found_problems

#%%
def scale_to_unit_interval(ndar, eps=1e-8):
	ndar = ndar.copy()
	ndar -= ndar.min()
	ndar *= 1.0 / (ndar.max() + eps)
	return ndar
#%%
def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)
#%%
def uniform(rng, shape, scale=0.08):
    return np.asarray(rng.uniform(
                    low=-scale, high=scale,
                    size=shape),
                    dtype=theano.config.floatX)
#%%
def zero(rng, shape):
    return np.asarray(np.zeros(shape),
                      dtype=theano.config.floatX)

#%%
def one(rng, shape):
    return np.asarray(np.ones(shape),
                      dtype=theano.config.floatX)

#%%
def eye(rng, rows):
    return np.asarray(np.eye(rows),
                      dtype=theano.config.floatX)

#%%
def orthogonal(rng, shape, scale=1.1, name=None):
    # From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = rng.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return scale * q[:shape[0], :shape[1]]

#%%
def get(rng, identifier, shape):

    if identifier == 'uniform':
        return uniform(rng,shape)
    elif identifier == 'orthonormal':
        return np.asarray(orthogonal(rng,shape), dtype=theano.config.floatX)
    elif identifier == 'zero':
        return zero(rng,shape)
    elif identifier == 'one':
        return one(rng,shape)
    elif identifier == 'eye':
        return eye(rng,shape[0])
    else:
        raise NotImplementedError