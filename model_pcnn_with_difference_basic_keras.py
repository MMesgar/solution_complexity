#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 11:23:58 2017

@author: mesgarmn
"""
import numpy as np
init_seed=123456
np.random.seed(init_seed)

import keras
from keras.layers import Dense, Embedding,Dropout,merge
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling1D, Convolution1D,Flatten,GlobalMaxPooling1D
from keras.layers.merge import Concatenate
import warnings
warnings.filterwarnings("ignore") 

#%%
def train_model_cnn(rng,
                    datasets,
                    U,
                    hidden_units, 
                    dropout_rate,
                    n_epochs, 
                    batch_size,
                    lr,
                    clip_value,
                    model_dir,
                    filter_sizes
                    ):
    
    
    Words = np.vstack([np.zeros(U.shape[1]),U])

    # load data
    (x_train,y_train), (x_valid,y_valid), (x_test,y_test) = load_data(rng,datasets)

    # define 3 parallel neural nets

    input_shape = (x_train.shape[1],)
    inputs = Input(shape=input_shape)
    from keras.layers import Lambda
    def get_x1(x):
        sent_w=x.shape[1]//3
        return x[:, :sent_w]
    def get_x2(x):
        sent_w=x.shape[1]//3
        return x[:, sent_w:2*sent_w]    
    def get_x3(x):
        sent_w=x.shape[1]//3
        return x[:, 2*sent_w:]
    def get_shape(x_shape):
        shape = list(x_shape)
        shape[-1]/=3
        return tuple(shape)
    inp1 = Lambda(get_x1,output_shape=get_shape)(inputs)
    inp2 = Lambda(get_x2,output_shape=get_shape)(inputs)
    inp3 = Lambda(get_x3,output_shape=get_shape)(inputs)


    n_symbols = Words.shape[0]
    embedding_size= Words.shape[1]#300

    emb1 = Embedding(output_dim=embedding_size, 
                        input_dim=n_symbols, 
                        mask_zero=False, # in CNN is False because we cannot give it to conv_layer
                        weights=[Words])(inp1)
    emb1 = Dropout(rate=dropout_rate[0],seed=init_seed)(emb1)

    emb2 = Embedding(output_dim=embedding_size, 
                        input_dim=n_symbols, 
                        mask_zero=False, 
                        weights=[Words])(inp2)
    emb2 = Dropout(rate=dropout_rate[0],seed=init_seed)(emb2)

    emb3 = Embedding(output_dim=embedding_size, 
                        input_dim=n_symbols, 
                        mask_zero=False, 
                        weights=[Words])(inp3)
    emb3 = Dropout(rate=dropout_rate[0],seed=init_seed)(emb3)

    conv_blocks1 = []
    for fs in filter_sizes:
        conv1 = Convolution1D(filters=hidden_units[0],
                             kernel_size=fs,
                             padding="valid",
                             activation="relu",
                             strides=1)(emb1)
        conv1 = GlobalMaxPooling1D()(conv1)
        
        conv_blocks1.append(conv1)
    CNN1 = Concatenate()(conv_blocks1) if len(conv_blocks1) > 1 else conv_blocks1[0]


    conv_blocks2 = []
    for fs in filter_sizes:
        conv2 = Convolution1D(filters=hidden_units[0],
                             kernel_size=fs,
                             padding="valid",
                             activation="relu",
                             strides=1)(emb2)
        conv2 = GlobalMaxPooling1D()(conv2)
        
        conv_blocks2.append(conv2)
    CNN2 = Concatenate()(conv_blocks2) if len(conv_blocks2) > 1 else conv_blocks2[0]


    conv_blocks3 = []
    for fs in filter_sizes:
        conv3 = Convolution1D(filters=hidden_units[0],
                             kernel_size=fs,
                             padding="valid",
                             activation="relu",
                             strides=1)(emb3)
        
        conv3 = GlobalMaxPooling1D()(conv3)
       
        conv_blocks3.append(conv3)
    CNN3 = Concatenate()(conv_blocks3) if len(conv_blocks3) > 1 else conv_blocks3[0]


    
    def get_subtract(x):
        return x[0]-x[1]
    def get_subtract_shape(input_shape):
        return input_shape[0]
    diff = Lambda(get_subtract,
                  output_shape=get_subtract_shape)\
                  ([CNN2,CNN3])
    
    out = merge([CNN1,CNN2,CNN3,diff], mode='concat', concat_axis=1)
    
    out = Dropout(rate = dropout_rate[1],seed=init_seed)(out)
    
    out  = Dense(units=hidden_units[1],activation='relu')(out)

    out = Dropout(rate = dropout_rate[2],seed=init_seed)(out)
    
    prediction = Dense(units=2,activation='softmax')(out)

    
    
    model = Model(inputs=inputs, output=prediction)

    model.compile(loss='categorical_crossentropy',
                   optimizer=keras.optimizers.Adadelta(lr=lr,
                                                       clipvalue=clip_value),
                   metrics=['accuracy'])


    from keras.callbacks import ModelCheckpoint
    model_chk_path = model_dir+'best_cnn.hdf5'
    mcp = ModelCheckpoint(model_chk_path, monitor="val_acc",
                      save_best_only=True, save_weights_only=True,mode='max')

    print('training ...')
    model.fit(x_train,
               y_train,
               batch_size=batch_size,
               epochs=n_epochs,
               validation_data=(x_valid, y_valid),
               shuffle=True,
               callbacks=[mcp]
               )

    print("evaluting on test set ...")
    model.load_weights(model_chk_path)
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(lr=lr,
                                                      clipvalue=clip_value),
                  metrics=['accuracy'])
    
    
    score, acc = model.evaluate(x_test,
                                 y_test,
                                 batch_size=batch_size)
    
    print('Test score:', score)
    print('Test accuracy:', acc)
    
    labels_prob =  model.predict(x_test)
    
    predicted_labels = np.argmax(labels_prob,axis=1)
    gold_labels = y_test.nonzero()[1]
    #compute the errors in the prediction
    
    errors= np.not_equal(predicted_labels,gold_labels)
    return acc,  errors,labels_prob


#    input_array  = x_train[0:2]
#    print input_array.shape
#    output_array = model1.predict(input_array)
#    print output_array

#%%
#load data
def load_data(rng, 
              datasets):
    input_train = datasets[0]
    input_valid = datasets[1]
    input_test = datasets[2]

    
    num_train_samples = input_train[0].shape[0]
    indices = rng.permutation(num_train_samples)
    input_train = (input_train[0][indices],input_train[1][indices])
     
    train_set_x = input_train[0]
    train_set_y = keras.utils.to_categorical(input_train[1],num_classes=2)
    
   
    val_set_x = input_valid[0]
    val_set_y = keras.utils.to_categorical(input_valid[1],num_classes=2)
    
    
    #build test_set
    test_set_x = input_test[0]
    test_set_y = keras.utils.to_categorical(input_test[1],num_classes=2)

    
    return (train_set_x,train_set_y), (val_set_x,val_set_y), (test_set_x, test_set_y)

#%%
import time
import sys
import cPickle
from evaluation import fold_output_evaluation,load_problem_solutions
from utils import make_idx_data_cv,print_errors_in_file
#sys.argv = ['',
#            '-nonstatic',
#            '-word2vec',
#            3,
#            2,
#            0.5,#drop_out
#            './data/corpus_2_3/',
#            './evalutions/nonstatic/',
#            600]
if __name__=="__main__":
    print "local start time :", time.asctime(time.localtime(time.time()) )
    # initialization
    sys.setrecursionlimit(10000)
    max_sent_len = int(sys.argv[8])
    print "max_sent_len = %d"%max_sent_len    
    
    in_dir = str(sys.argv[6])
    dataset_file = in_dir+'dataset_triplet.p'
    print 'dataset_file :%s'%dataset_file
    mode= sys.argv[1]
    word_vectors =  sys.argv[2]
    rng = np.random.RandomState(23455)
    
	
    # load dataset
    print "loading data...",
    x = cPickle.load(open(dataset_file,"rb"))
    ds, W, W2, word_idx_map, vocab, folds = x[0], x[1], x[2], x[3], x[4], x[5]
    print "data loaded!"
    
    #process input arguments
    if mode=="-nonstatic":
        print "model architecture: CNN-non-static"
        non_static=True
    elif mode=="-static":
        print "model architecture: CNN-static"
        non_static=False
    execfile("conv_net_classes.py")    
    if word_vectors=="-rand":
        print "using: random vectors"
        U = W2
    elif word_vectors=="-word2vec":
        print "using: word2vec vectors"
        U = W   
    n_epochs = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    dropout_rate= float(sys.argv[5])
    output_dir= str(sys.argv[7])


    #load problem_solution; we need it for kendall tau computation
    corpus_path = in_dir+'corpus.txt'
    problem_solutions = load_problem_solutions(corpus_path, clean_string= True)

    # keep performance of each fold in results
    folds_acc = []
    folds_tau = []
    folds_rank1_acc = []
    folds_rankn_acc = []
    folds_rank_both_acc= []
  
    

    # iterate over each fold
    for i, fold in enumerate(folds):
        error_file_path = ''
        
        #construct train and test sets
        datasets, test_dict = make_idx_data_cv(rng, ds, word_idx_map, fold,
                                    max_l=max_sent_len,  filter_h=5)
    
        print "processing fold = %d"%i
        print "number of problems in train_set: %d in valid_set: %d and test_Set: %d"%(len(fold[0]), len(fold[1]), len(fold[2]))
        print "number of training samples (triplets) =%d"%datasets[0][0].shape[0]
        print "number of validation samples (triplets) = %d"%datasets[1][0].shape[0]
        print "number of test samples (triplets)= %d"%datasets[2][0].shape[0]
        print "number of samples testset (problem_id)=%d"%len(test_dict.keys())
        
        perf, errors,labels_prob = \
        train_model_cnn(rng,
                        datasets,
                        U,
                        hidden_units=[1000,#CNN output size
                                      1000,#units in HL
                                      2], # units in output layer
                        n_epochs=n_epochs, 
                        batch_size=batch_size,
                        lr= 0.01,
                        clip_value=9,
                        dropout_rate=[0.0,#embeddings
                                      dropout_rate,#CNN after merge
                                      dropout_rate#HL
                                      ],
                        model_dir='./models/',
                        filter_sizes=[4,5]
                        )
                              
            
        print "test perf:%f%% "%(perf*100)
        folds_acc.append(perf)
        error_file_path += (output_dir+'fold'+str(i)+'.error')
        found_problems = print_errors_in_file(labels_prob, errors, datasets[2], test_dict, word_idx_map, error_file_path)
        print('number of found_problems in error:%d'%len(found_problems))
        
        fold_tau, results, acc = fold_output_evaluation(fold_path=error_file_path,
                                             prob_solutions=problem_solutions)
        print("results: %s"%results)
        print("test tau: %f "%fold_tau)
        print("test accuracies (rank1, rankn, rank_both) = ", acc)
        folds_tau.append(fold_tau)
        folds_rank1_acc.append(acc[0])
        folds_rankn_acc.append(acc[1])
        folds_rank_both_acc.append(acc[2])

        
    # compute the final results
    print('The final accuracy is: %f'%np.mean(folds_acc))
    print('The final tau is: %f'%np.mean(folds_tau))
    print('The final rank_1 accuracy is: %f'%(np.mean(folds_rank1_acc)))
    print('The final rank_n accuracy is: %f'%(np.mean(folds_rankn_acc)))
    print('The final rank_both accuracy is: %f'%(np.mean(folds_rank_both_acc)))
    print "local end time :", time.asctime(time.localtime(time.time()) )
    print('Done')
