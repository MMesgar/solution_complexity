"""
Created on Sun Jan 15 19:55:03 2017

@author: Mohsen Mesgar
process the dataset and predict the labels
"""
"""
This file is sentence-pair model.
    
"""
#%%
import theano
import numpy as np
import cPickle
import sys
import warnings
import time
warnings.filterwarnings("ignore") 
import theano.tensor as T
from optimizers import sgd_updates_adadelta, SGD, AdaGrad
import activations
from evaluation import fold_output_evaluation,load_problem_solutions
from collections import defaultdict
rng = np.random.RandomState(seed = 9834)
from utils import debug_print,shared_dataset,get_idx_from_sent,make_idx_data_cv,print_errors_in_file
from utils import uniform, zero, one, eye, orthogonal, get
#%%
class RNN():
    def __init__(self,
                 rng,
                 input,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 activation=T.tanh,
                 init='uniform',
                 inner_init='orthonormal',
                 output_mean= True # True: mean False: last
                 ):
        
        self.input_dim = input_dim
        self.hidden_dim= hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.output_mean = output_mean
        
        
        #The input has the shape of (#batches, 1, #words = sent_h, emb_len=sent_w)
        #we should remove dimension 1 and get (#bacthes, #words, emb_len)
        self.input = input.dimshuffle(0,2,3)
        
        # we would like to iterate over rows (or words) of the matrix (or sentence).
        # so we reshuffle it again to have corresponding dim to the words at the first dim
        # output shape is (#words, batch_size, emb)
        self.input = self.input.dimshuffle(1, 0, 2)
        
        #initialize parameters of the model 
        self.W = theano.shared(value=get(rng,identifier=init, shape=(self.input_dim, self.hidden_dim)),
                               name='W',
                               borrow=True
                               )
        self.U = theano.shared(value=get(rng,identifier=inner_init, shape=(self.hidden_dim, self.hidden_dim)),
                               name='U',
                               borrow=True
                               )
        self.V = theano.shared(value=get(rng,identifier=init, shape=(self.hidden_dim, self.output_dim)),
                               name='V',
                               borrow=True
                               )
        self.bh = theano.shared(value=get(rng,identifier='zero', shape=(self.hidden_dim, )),
                                name='bh',
                                borrow=True)
        self.by = theano.shared(value=get(rng,identifier='zero', shape=(self.output_dim, )),
                                name='by',
                                borrow=True)

        self.h0 = theano.shared(value=get(rng,identifier='zero', shape=(self.hidden_dim, )), name='h0', borrow=True)
        
        
        self.params = [self.W, self.U, self.V, self.bh, self.by]
        
        # now we define step of recurrent input is input vector and last state
        def recurrence(x_t, h_tm_prev):
            #h_t is state
            h_t = self.activation(T.dot(x_t, self.W) + T.dot(h_tm_prev, self.U) + self.bh)
            
            #y_t is output
            y_t = T.nnet.softmax(T.dot(h_t, self.V) + self.by)
            
            return h_t, y_t
        
        # lets iterate over all words
        [self.h_t, self.y_t], _ = theano.scan(
            recurrence,
            sequences=self.input, # by default it loops over the first dim which is words
            outputs_info=[T.alloc(self.h0, self.input.shape[1], self.hidden_dim), None] # initializtion
        )
        
        # bring back the batch dim as the first one and  #words to the second dim
        # output has shape (#bacth, #words, hidden_dim)
        self.h_t = self.h_t.dimshuffle(1, 0, 2)
        self.y_t = self.y_t.dimshuffle(1, 0, 2)
        
        # output can be mean over all states or the last state
        # the dimension of the output state should be (batch_size, hidden_dim)
        self.last_h_t = self.h_t[:,-1,:]
        self.mean_h_t = T.mean(self.h_t,axis=1,keepdims=False)
        self.output = self.last_h_t
        if output_mean:
            self.output = self.mean_h_t
        
        #since we only use h_t to compute cost functions, we don't need parameters of y_t
        # if we leave them in the self.params, gradient function of theano gives an error
        self. params = self.params = [self.W, self.U, self.bh]
    
    
    def predict(self, new_data):
        """
        predict for new data. It generates the output of this RNN layer given new_data as the input

        """
        #The input has the shape of (batch_size, 1, #words = sent_h, emb_len=sent_w)
        #we should remove dimension 1 and get (batch_size, #words, emb_len)
        new_data = new_data.dimshuffle(0,2,3)
        
        # we would like to iterate over rows (or words) of the matrix (or sentence).
        # so we reshuffle it again to have corresponding dim to the words at the first dim
        # output shape is (#words, batch_size, emb)
        new_data = new_data.dimshuffle(1, 0, 2)
        
        # we define recurrence over new data.
        def recurrence(x_t, h_tm_prev):
            #h_t is state
            _h_t = self.activation(T.dot(x_t, self.W) + T.dot(h_tm_prev, self.U) + self.bh)
            
            #y_t is output
            _y_t = T.nnet.softmax(T.dot(_h_t, self.V) + self.by)
            
            return _h_t, _y_t
        
        # lets loop over all words
        [h_t, y_t], _ = theano.scan(
            recurrence,
            sequences=new_data, # by default it loops over the first dim which is words
            outputs_info=[T.alloc(self.h0, new_data.shape[1], self.hidden_dim), None] # initializtion
        )
        
        # bring back the batch dim as the first one and  #words to the second dim
        # final_dim is (#batch,#words, hid_dim)
        h_t = h_t.dimshuffle(1, 0, 2)
        y_t = y_t.dimshuffle(1, 0, 2)
        
        # output can be mean over all states or last state
        # the dimension of the output state should be (batch_size, hidden_dim)
        last_h_t = h_t[:,-1,:]
        mean_h_t = T.mean(h_t,axis=1,keepdims=False)
        output = last_h_t
        if self.output_mean:
            output = mean_h_t
        return output

#%%
def train_model_rnn(rng,
                    datasets,
                    U,
                    sent_w=300, 
                    hidden_units=[100,1], 
                    dropout_rate=[0.5],
                    shuffle_batch=True,
                    n_epochs=25, 
                    batch_size=2, 
                    lr_decay = 0.95,
                    activations=[activations.Iden],
                    sqr_norm_lim=9,
                    non_static=True,
                    num_cells=100
                    ):
    
    input_train = datasets[0]
    input_valid = datasets[1]
    input_test = datasets[2]
    
    # sent_w is equal to the embedding size
    sent_w = sent_w
    
    # sent_h is one third of the length of each instance, because
    # we concatenated problem and solution vectors
    sent_h = (len(input_train[0][0]))/3 # length of sentence
    
        
    # index to a [mini]batch
    index = T.lscalar()  
    
    # x is problem + solution vector and y is label
    x = T.matrix('x')  
    y = T.ivector('y')  
    
    # create a zero vector for zero padding
    Words = theano.shared(value = U, name = "Words")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(sent_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))], allow_input_downcast=True)
    
    # extract problem and solution
    # x1 as problem and x2 as positive solution (less complex) and x3 as a negative solution
    x1 = debug_print(x[:,:sent_h],"x1", False)
    x2 = debug_print(x[:,sent_h:2*sent_h],"x2", False)
    x3 = debug_print(x[:,2*sent_h:],"x3", False)
    
    
    # replace word indecides with word vectors
    layer0_input1 = Words[T.cast(x1.flatten(),dtype="int32")].reshape((x1.shape[0],1,x1.shape[1],Words.shape[1]))#Words.shape[1]=300, x.shape[1]= sentence length                                  
    layer0_input2 = Words[T.cast(x2.flatten(),dtype="int32")].reshape((x2.shape[0],1,x2.shape[1],Words.shape[1]))
    layer0_input3 = Words[T.cast(x3.flatten(),dtype="int32")].reshape((x3.shape[0],1,x3.shape[1],Words.shape[1]))
    
    
    # three parallel layers: one for processing problem and
    # one for processing solution2
    # one for processing solution3
    
    # Define a RNN layer vanilla cell; the output should be in this shape 
    # (batch_size,output_size) = (32,100)
    hidden_dim = hidden_units[0]
    output_dim = hidden_units[-1]
    rnn_layer1 = RNN(rng, 
                   input = layer0_input1,
                   input_dim = sent_w, # emb_dim or sent_h (only for one step)
                    output_dim = output_dim,# output_dim
                    hidden_dim= hidden_dim #hidden_state_dim,
                   )
    
    rnn_layer2 = RNN(rng, 
                   input = layer0_input2,
                   input_dim = sent_w, 
                    output_dim = output_dim,# output_dim
                    hidden_dim= hidden_dim #hidden_state_dim,
                    )
    
    rnn_layer3 = RNN(rng, 
                   input = layer0_input3,
                   input_dim = sent_w, 
                   output_dim = output_dim,# output_dim
                    hidden_dim= hidden_dim #hidden_state_dim,
                    )
    
    layer1_input1= rnn_layer1.output
    layer1_input2= rnn_layer2.output
    layer1_input3= rnn_layer3.output


    difference  = layer1_input3 - layer1_input2
       
    layer1_input = T.concatenate([layer1_input1, layer1_input2, layer1_input3,difference],1) 

    
    # now we concatenate them parts of layer1_input to build the its input vector
    #shape = (bacth_size, hidden_dim+hidden_dim  +hidden_dim +hidden_dim) = (32,400)
    layer1_input = T.concatenate([layer1_input1,layer1_input2,layer1_input3],1)
    
    hidden_units[0] = hidden_dim + hidden_dim + hidden_dim + hidden_dim  
    
    # now we give the output of this  convolution and pooling layer to
    # a dropout layer
    classifier = MLPDropout(rng, 
                            input=layer1_input,
                            layer_sizes=hidden_units,
                            activations=activations,
                            dropout_rates=dropout_rate)

    
    #define parameters of the model and update functions using adadelta
    params = classifier.params + \
            rnn_layer1.params + \
            rnn_layer2.params +\
            rnn_layer3.params 

    
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]
    
    # cost function
    cost = classifier.negative_log_likelihood(y) 
    dropout_cost = classifier.dropout_negative_log_likelihood(y) 
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)

    
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate 
    #extra data (at random)
    # datasets[0] is train set
    num_train_samples = input_train[0].shape[0]
    if num_train_samples % batch_size > 0:
        extra_data_num = batch_size - num_train_samples % batch_size
        indices = rng.permutation(num_train_samples)
        train_set_x = input_train[0][indices]
        train_set_y = input_train[1][indices]
        extra_data_x = train_set_x[:extra_data_num]
        extra_data_y = train_set_y[:extra_data_num]
        new_data=(np.append(input_train[0],extra_data_x,axis=0),np.append(input_train[1],extra_data_y,axis=0))
    else:
        new_data = input_train
    
    num_train_samples = new_data[0].shape[0]
    indices = rng.permutation(num_train_samples)
    new_data = (new_data[0][indices],new_data[1][indices])
    
    n_batches = num_train_samples/batch_size
    
    # pick up 90% of the training data for the training
    n_train_batches = n_batches
     
    #divide train set into train/val sets 
    train_set_x = new_data[0]
    train_set_y = new_data[1]
 
    val_set_x = input_valid[0] 
    val_set_y = input_valid[1]
    n_val_batches = val_set_x.shape[0] / batch_size
    
    #build test_set
    test_set_x = input_test[0] 
    test_set_y = np.asarray(input_test[1],"int")

    
    # convert train and val  set to shared data            
    train_set_x, train_set_y = shared_dataset((train_set_x,train_set_y))
    val_set_x, val_set_y = shared_dataset((val_set_x,val_set_y))
    
   
   
    
    # define validation model
    val_model = theano.function([index], classifier.errors(y),
         givens={
            x: val_set_x[index * batch_size: (index + 1) * batch_size],
             y: val_set_y[index * batch_size: (index + 1) * batch_size]},
                                allow_input_downcast=True)
    
    
    #define test model 
    #Note: I use train_set to get error on the training data
    # we run the model on the test data some lines further
    test_model = theano.function([index], classifier.errors(y),
             givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                 y: train_set_y[index * batch_size: (index + 1) * batch_size]},
                                 allow_input_downcast=True)               

     #Train model, to check the cost over the training data
    train_model = theano.function([index], cost, updates=grad_updates,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
              y: train_set_y[index*batch_size:(index+1)*batch_size]},
                                  allow_input_downcast = True) 

        
    
    #compute the number of test samples
    test_size = test_set_x.shape[0]
    
    # replace word indecies with word vectors
    test_layer0_input1 = Words[T.cast(x1.flatten(),dtype="int32")].reshape((test_size,1,sent_h,Words.shape[1]))
    test_layer0_input2 = Words[T.cast(x2.flatten(),dtype="int32")].reshape((test_size,1,sent_h,Words.shape[1]))
    test_layer0_input3 = Words[T.cast(x3.flatten(),dtype="int32")].reshape((test_size,1,sent_h,Words.shape[1]))
    
    test_layer0_output1 = rnn_layer1.predict(test_layer0_input1)
    test_layer0_output2 = rnn_layer2.predict(test_layer0_input2)
    test_layer0_output3 = rnn_layer3.predict(test_layer0_input3)

    difference = test_layer0_output3  -  test_layer0_output2



    # conclude the inputs of the next layer
    test_layer1_input = T.concatenate([test_layer0_output1,
                                       test_layer0_output2,
                                       test_layer0_output3,
                                       difference],1)
    
    
    # compute the score on the testset
    test_y_pred = classifier.predict(test_layer1_input)
    
    # get label probabilities
    labels_probs = classifier.predict_p(test_layer1_input)

    #compute the errors in the prediction
    error_vector = T.neq(test_y_pred,y)

    #compute the test error
    test_error = T.mean(error_vector)
    
    # test model
    test_model_all = theano.function([x,y], [test_error, error_vector, labels_probs] , allow_input_downcast = True)   

    #start training over mini-batches
    print '... training'
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0       
    cost_epoch = 0    
    while (epoch < n_epochs):
        start_time = time.time()
        epoch = epoch + 1
        if shuffle_batch:
            for minibatch_index in rng.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)  
                set_zero(zero_vec)
        
        train_losses = [test_model(i) for i in xrange(n_train_batches)]
        train_perf = 1 - np.mean(train_losses)
        val_losses = [val_model(i) for i in xrange(n_val_batches)]
        val_perf = 1- np.mean(val_losses)                        
        print('epoch: %i, training time: %.2f secs, train perf: %.2f %%, val perf: %.2f %%' % (epoch, time.time()-start_time, train_perf * 100., val_perf*100.))
        if val_perf >= best_val_perf:
            best_val_perf = val_perf
            test_error, errors, labels_probs = test_model_all(test_set_x,test_set_y)        
            test_perf = 1 - test_error
    return test_perf, errors, labels_probs

#%%
#sys.argv = ['','-nonstatic','-word2vec',1,2,0.5,'./data/corpus_all/','./evalutions/nonstatic/']
if __name__=="__main__":
    print "local start time :", time.asctime(time.localtime(time.time()) )
    # initialization
    sys.setrecursionlimit(10000)
    theano.config.on_unused_input ='ignore' 
    theano.config.floatX = 'float32'
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
        filter_sizes= [4,5]

        #construct train and test sets
        datasets, test_dict = make_idx_data_cv(rng, ds, word_idx_map, fold,
                                    max_l=max_sent_len,filter_h=np.max(filter_sizes))
    
        print "processing fold = %d"%i
        print "number of problems in train_set: %d in valid_set: %d and test_Set: %d"%(len(fold[0]), len(fold[1]), len(fold[2]))
        print "number of training samples (triplets) =%d"%datasets[0][0].shape[0]
        print "number of validation samples (triplets) = %d"%datasets[1][0].shape[0]
        print "number of test samples (triplets)= %d"%datasets[2][0].shape[0]
        print "number of samples testset (problem_id)%d"%len(test_dict.keys())
        
        perf, errors,labels_prob = train_model_rnn(rng,
                              datasets,
                              U,
                              lr_decay=0.9,
                              #hidden_units is tricky:
                                  # in CNN:
                                      # hidden_units[0] is number of feature_maps
                                      # hidden_units[-1] is number of class labels
                                  # in RNN: 
                                      #hidden_units[0] is number of nodes in rnn_hidden_layer
                                      #hidden_units[-1] is number of class lables
                              hidden_units=[1000,1000,2], # 
                              shuffle_batch=True, 
                              n_epochs=n_epochs, 
                              sqr_norm_lim=9.0,
                              non_static=non_static,
                              batch_size=batch_size,
                              activations=[activations.Iden,
                                           activations.ReLU,
                                           activations.ReLU],
                              dropout_rate=[dropout_rate,
                                            dropout_rate,
                                            dropout_rate])
                              
            
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



