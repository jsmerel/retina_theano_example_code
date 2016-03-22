"""
Code is generally based on the MLP tutorial code for theano

There are a bunch of classes at the top of the code (for individual layers of the network), then the main function declares and trains the architecture.

"""
__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import time
import pdb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *

import numpy
import h5py

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.tensor.nnet import conv3d2d
from sys import exit

import copy

######################
######################
# Adam #
######################
######################


#Copyright (c) 2015 Alec Radford
def Adam(cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
    updates = []
    grads = T.grad(cost, params)
    i = theano.shared(numpy.asarray(0., dtype=theano.config.floatX), borrow=True)
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates

######################
######################
# LOAD DATA FUNCTION #
######################
######################

"""
Loads in the prepared dataset
stimuli are images and responses are neural activity of parasol cells

"""

def load_data(exp,on_ind):
    ''' Loads the dataset
    '''

    #############
    # LOAD DATA #
    #############
    print '... loading data'
    
    #basedir = '/vega/stats/users/erb2180/RetinaProject/Data/'
    basedir = '/Volumes/Backup/RetinaProject/Data/Data/'
    dataset= ''+basedir+'NSEM'+exp+'_movResp_crop.mat'
    
    
    f = h5py.File(dataset)
    stimuli_train = numpy.array(f['FitMovies'])/255.00
    responses_train2 = numpy.array(f['FitResponses'])
    stimuli_test = numpy.array(f['TestMovie'])/255.00
    responses_test2 = numpy.array(f['TestResponses'])
    trainBatchSize = numpy.array(f['fit_batch'])
    testBatchSize = numpy.array(f['test_batch'])
    RGC_locations2 = numpy.array(f['locations'])
    temporalFilter = numpy.array(f['tempFilt'])
    cell_ind = numpy.array(f['ONcell_vec'])
    f.close()
  
    # Pad the image so RF centers close to the edge won't matter
    stimuli_train = numpy.concatenate([stimuli_train, numpy.mean(stimuli_train)*numpy.ones((stimuli_train.shape[0],80))],axis=1) #,numpy.zeros((stimuli_train.shape[0],80))],axis=1)
    stimuli_test = numpy.concatenate([stimuli_test, numpy.mean(stimuli_train)*numpy.ones((stimuli_test.shape[0],80))],axis=1) #,numpy.zeros((stimuli_test.shape[0],80))],axis=1)
    #stimuli_train = numpy.concatenate([stimuli_train, numpy.zeros((stimuli_train.shape[0],80))],axis=1) #,numpy.zeros((stimuli_train.shape[0],80))],axis=1)
    #stimuli_test = numpy.concatenate([stimuli_test, numpy.zeros((stimuli_test.shape[0],80))],axis=1) #,numpy.zeros((stimuli_test.shape[0],80))],axis=1)
    
    stimuli_train = stimuli_train.astype('float32')

    # Normalization methods
    
    # Subtract mean from every example
    #stimuli_train = stimuli_train - numpy.reshape(numpy.mean(stimuli_train,axis=1),(stimuli_train.shape[0],1))
    #stimuli_test = stimuli_test - numpy.reshape(numpy.mean(stimuli_test,axis=1),(stimuli_test.shape[0],1))
    
    # Feature standardization
    
    #mean_train = numpy.reshape(numpy.mean(stimuli_train,axis=0),(1,stimuli_train.shape[1]))
    #stimuli_train = stimuli_train - mean_train
    #stimuli_test = stimuli_test - mean_train
    
    #std_train = numpy.std(stimuli_train,axis=0)
    #stimuli_train = stimuli_train / std_train
    #stimuli_test = stimuli_test / std_train

    # Separate out the ON and OFF cells
    responses_test = responses_test2[:,cell_ind[0,:]==on_ind]
    responses_train = responses_train2[:,cell_ind[0,:]==on_ind]
    RGC_locations = RGC_locations2[:,cell_ind[0,:]==on_ind]

    data_set_train = (stimuli_train, responses_train)
    data_set_test = (stimuli_test, responses_test)

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=True)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=True)

        return shared_x, shared_y
    
    data_set_x_train, data_set_y_train = shared_dataset(data_set_train)
    data_set_x_test, data_set_y_test = shared_dataset(data_set_test)
    rval = [(data_set_x_train, data_set_y_train), (data_set_x_test, data_set_y_test), trainBatchSize, testBatchSize, RGC_locations, temporalFilter]

    return rval

######################
######################
# Layer Classes #
######################
######################

class HiddenLayerRNN1(object):
    def __init__(self, rng, input, n_in, n_out, Ncells,ordered_rgc_indices):
        """
            RNN hidden layer: units are fully-connected and have
            a rectified linear activation. Weights project inputs to the units which are recurrently connected.
            Weight matrix W is of shape (n_in,n_out)
            and the bias vector b is of shape (n_out,).
            
            :type rng: numpy.random.RandomState
            :param rng: a random number generator used to initialize weights
            
            :type input: theano.tensor.dmatrix
            :param input: a symbolic tensor of shape (n_examples, n_in)
            
            :type n_in: int
            :param n_in: dimensionality of input
            
            :type n_out: int
            :param n_out: number of hidden units

            """
        self.input = input


        # Initialize weights/biases
        W_values = numpy.asarray(rng.uniform(
            low=-.2*sqrt(6.00 / (n_in + n_out)),
            high=.2*sqrt(6.00 / (n_in + n_out)),
                                             size=(n_in, n_out)), dtype=theano.config.floatX)
        W = theano.shared(value=W_values, name='W', borrow=True)
        
        b_values = 1e-14*numpy.ones((n_out,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, name='b', borrow=True)
        
        Q = numpy.random.randn(n_out, n_out).astype(theano.config.floatX)
        W_RNN_values, s, v = numpy.linalg.svd(Q)
        W_RNN = theano.shared(value=W_RNN_values, name='W_RNN', borrow=True)

        # parameters of this layer
        self.W = W
        self.b = b
        self.W_RNN = W_RNN
        self.params = [self.W_RNN, self.W, self.b]

        #initial hidden state values
        h_0 = T.zeros((Ncells,n_out))
  
        # recurrent function with rectified linear output activation function (u is input, h is hidden activity)
        def step(u_t, h_tm1):
            u_t_new = u_t[ordered_rgc_indices]
            lin_E = T.dot(u_t_new, self.W) + T.dot(h_tm1, self.W_RNN) + self.b
            h_t = lin_E*(lin_E>0)
            return h_t

        # compute the timeseries
        h, _ = theano.scan(step,
                   sequences=self.input,
                   outputs_info=h_0)
            
        # output activity is the hidden unit activity
        self.output = h

class HiddenLayerRNN(object):
    def __init__(self, rng, input, n_in, n_out, Ncells):
        """
            RNN hidden layer: units are fully-connected and have
            a rectified linear activation. Weights project inputs to the units which are recurrently connected.
            Weight matrix W is of shape (n_in,n_out)
            and the bias vector b is of shape (n_out,).
            
            :type rng: numpy.random.RandomState
            :param rng: a random number generator used to initialize weights
            
            :type input: theano.tensor.dmatrix
            :param input: a symbolic tensor of shape (n_examples, n_in)
            
            :type n_in: int
            :param n_in: dimensionality of input
            
            :type n_out: int
            :param n_out: number of hidden units

            """
        self.input = input


        # Initialize weights/biases
        W_values = numpy.asarray(rng.uniform(
            low=-.2*sqrt(6.00 / (n_in + n_out)),
            high=.2*sqrt(6.00 / (n_in + n_out)),
                                             size=(n_in, n_out)), dtype=theano.config.floatX)
        W = theano.shared(value=W_values, name='W', borrow=True)
        
        b_values = 1e-14*numpy.ones((n_out,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, name='b', borrow=True)
        
        Q = numpy.random.randn(n_out, n_out).astype(theano.config.floatX)
        W_RNN_values, s, v = numpy.linalg.svd(Q)
        W_RNN = theano.shared(value=W_RNN_values, name='W_RNN', borrow=True)

        # parameters of this layer
        self.W = W
        self.b = b
        self.W_RNN = W_RNN
        self.params = [self.W_RNN, self.W, self.b]

        #initial hidden state values
        h_0 = T.zeros((Ncells,n_out))
  
        # recurrent function with rectified linear output activation function (u is input, h is hidden activity)
        def step(u_t, h_tm1):
            lin_E = T.dot(u_t, self.W) + T.dot(h_tm1, self.W_RNN) + self.b
            h_t = lin_E*(lin_E>0)
            return h_t

        # compute the timeseries
        h, _ = theano.scan(step,
                   sequences=self.input,
                   outputs_info=h_0)
            
        # output activity is the hidden unit activity
        self.output = h

class PoissonRegression_alln(object):
    """Poisson Regression Class

    The poisson regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. 
    """

    def __init__(self, input, n_in, n_out, obj_function, neurnum):
        """ Initialize the parameters of the poisson regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        W = 1e-14*numpy.ones((n_out,n_in), dtype=theano.config.floatX)
        self.W = theano.shared(value=W,
                                name='W_poiss', borrow=True)
        b_values = -2*numpy.ones((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b_poiss', borrow=True)

        # compute vector of expected values (for each output) in symbolic form

        self.E_y_given_x = T.log(1 + T.exp(T.sum(input*self.W,axis=2) + self.b))
        
        # parameters of the model
        self.params = [self.W, self.b]

        self.obj_f_type = obj_function

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::
        p(y_observed|model,x_input) = E[y|x]^y exp(-E[y|x])/factorial(y)
        
        take sum over output neurons and times

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        """
        
        return -T.sum((  (y * T.log(self.E_y_given_x)) - (self.E_y_given_x)  ) , axis = 0)
    
    def negative_log_likelihood_trialaverageFit(self, y, ntrials):
        """Return the mean of the negative log-likelihood of the prediction
            of this model under a given target distribution.
            
            .. math::
            p(y_observed|model,x_input) = E[y|x]^y exp(-E[y|x])/factorial(y)
            
            take sum over output neurons and times
            
            :type y: theano.tensor.TensorType
            :param y: corresponds to a vector that gives for each example the
            correct label
            
            """
        return -T.sum((  (y[120:] * T.log(self.E_y_given_x[120:])) - ntrials*(self.E_y_given_x[120:])  ) , axis = 0)

######################
######################
# Main function    #
######################
######################

def SGD_training():

    exp_num = int(sys.argv[1]) # which of 4 experiments should be used
    on_ind = int(sys.argv[2]) # ON vs OFF cells, 1 for ON

    # Decide which experiment/cell type to use
    if exp_num == 1:
        exp = '2012-08-09-3'
    elif exp_num == 2:
        exp = '2012-09-27-3'
    elif exp_num == 3:
        exp = '2013-08-19-6'
    elif exp_num == 4:
        exp = '2013-10-10-0'

    if on_ind == 1:
        cell_type = 'ON'
    elif on_ind == 0:
        cell_type = 'OFF'

    # Load data
    dataset_info = load_data(exp,on_ind)

    data_set_x_train, data_set_y_train = dataset_info[0]
    data_set_x_test, data_set_y_test = dataset_info[1]

    trainBatchSize = int(dataset_info[2][0][0])
    testBatchSize = int(dataset_info[3][0][0])
    batch_size = min(trainBatchSize,testBatchSize) #train and test batch sizes should be same size
    RGC_locations = dataset_info[4]
    temporalFilter = dataset_info[5]

    # Number of batches/indices of movies
    totalTrainBatches = 118
    n_valid_batches = 10
    n_train_batches = totalTrainBatches - n_valid_batches #(data_set_y_train.get_value(borrow=True).shape[0])/batch_size
    n_test_batches = (data_set_y_test.get_value(borrow=True).shape[0])/testBatchSize
    Ncells = data_set_y_train.shape[1].eval()
    #valInds = numpy.random.choice(totalTrainBatches,10,replace=False) # Choose validation movies randomly from the training movies
    allInds = numpy.arange(0,totalTrainBatches)
    valInds = numpy.arange(5,118,12) # Use same validation movies every time
    trainInds = numpy.delete(allInds,valInds) # Use all training movies except those used for validation

    # Get indices of image patches (to swap out neurons)
    ordered_rgc_indices2 = numpy.zeros((Ncells,31*31))
    for i in xrange(0,Ncells):
        #print i
        image_patch_size = [30, 30]
        yinds = [RGC_locations[0,i].astype(int)-image_patch_size[0]/2,RGC_locations[0,i].astype(int)+image_patch_size[0]/2+1]
        xinds = [RGC_locations[1,i].astype(int)-image_patch_size[1]/2,RGC_locations[1,i].astype(int)+image_patch_size[1]/2+1]
        flatInds = numpy.arange(0,3200,1).reshape(40,80)
        flatInds = numpy.concatenate([flatInds,3201*numpy.ones((20,80))])
        flatInds = numpy.concatenate([flatInds,3201*numpy.ones((60,20))],axis=1)
        theseInds = flatInds[max(yinds[0]-1,0):yinds[1]-1,max(xinds[0]-1,0):xinds[1]-1]
        if theseInds.shape[0]<(image_patch_size[0]+1):
            theseInds = numpy.concatenate([3201*numpy.ones(((image_patch_size[0]+1)-theseInds.shape[0],theseInds.shape[1])),theseInds])
        if theseInds.shape[1]<(image_patch_size[1]+1):
            theseInds = numpy.concatenate([3201*numpy.ones((theseInds.shape[0],(image_patch_size[1]+1)-theseInds.shape[1])),theseInds],axis=1)
        
        hh = theseInds.flatten()
        ordered_rgc_indices2[i,:] = hh

    ordered_rgc_indices= theano.shared(numpy.asarray(ordered_rgc_indices2,dtype=int64),borrow=True)

    # Create data summed over test trials
    data_set_y_test_alltrials = theano.shared(value=numpy.zeros((batch_size,Ncells),dtype=theano.config.floatX),borrow=True)
    for i_trial in xrange(0,n_test_batches):
        data_set_y_test_alltrials += data_set_y_test[i_trial * batch_size:(i_trial+ 1) * batch_size,:]

    # Create new data structure
    #data_set_x_test2 = theano.shared(value=numpy.zeros((data_set_x_test.shape[0].eval(),Ncells,961),dtype=theano.config.floatX),borrow=True)
    # data_set_x_train2 = theano.shared(value=numpy.zeros((data_set_x_train.shape[0].eval(),Ncells,961),dtype=theano.config.floatX),borrow=True)
    # for neur in xrange(0,Ncells):
    #     data_set_x_test2 = T.set_subtensor(data_set_x_test2[:,neur,:],data_set_x_test[:,ordered_rgc_indices[neur,:]])
    #    data_set_x_train2 = T.set_subtensor(data_set_x_train2[:,neur,:],data_set_x_train[:,ordered_rgc_indices[neur,:]])


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    i_n = T.lscalar()
    x = T.matrix('x')  # the data is presented as a vector of inputs with many exchangeable examples of this vector
    y = T.matrix('y')  # the output is a vector of matched output unit responses.
    neurnum = T.matrix('neurnum')
    rng = numpy.random.RandomState(1234)

    ################
    # Architecture #
    ################

    Layer2 = HiddenLayerRNN1(rng,input=x,n_in=31*31,n_out=50,Ncells=Ncells,ordered_rgc_indices=ordered_rgc_indices)
    Layer3 = HiddenLayerRNN(rng,input=Layer2.output,n_in=50,n_out=50,Ncells=Ncells)
    Layer4 = PoissonRegression_alln(input=Layer3.output,n_in=50,n_out=Ncells,obj_function='poiss',neurnum=i_n)


    #######################
    # Objective function
    #######################

    negative_log_likelihood = Layer4.negative_log_likelihood


    #######################
    # Architecture params #
    #######################

    params = Layer4.params + Layer3.params + Layer2.params


    # the cost we minimize during training is the negative log likelihood of
    # the model plus possibly the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = T.sum(negative_log_likelihood(y))

    ##########################
    # Create model functions #
    ##########################

#    test_model = theano.function(inputs=[index,i_n],
#            outputs=[negative_log_likelihood(y), Layer4.E_y_given_x, y],
#            givens={
#                x: data_set_x_test[0 * testBatchSize:(0 + 1) * batch_size, ordered_rgc_indices[i_n,:]],
#                y: data_set_y_test[index * testBatchSize:(index + 1) * batch_size,i_n],
#                neurnum: i_n})

    test_model = theano.function(inputs=[],
            outputs=[Layer4.negative_log_likelihood_trialaverageFit(y,n_test_batches), Layer4.E_y_given_x, y],
            givens={
                x: data_set_x_test[0 * testBatchSize:(0 + 1) * batch_size, :],
                y: data_set_y_test_alltrials,
                neurnum: i_n})

    validate_model = theano.function(inputs=[index],
            outputs=cost,
            givens={
                x: data_set_x_train[index * batch_size:(index + 1) * batch_size,:],
                y: data_set_y_train[index * batch_size:(index + 1) * batch_size,:],
                neurnum: i_n})

    # Specify updates to params
    updates = Adam(cost, params)

    train_model = theano.function(inputs=[index], outputs=cost,
            updates=updates,
            givens={
                x: data_set_x_train[index * batch_size:(index + 1) * batch_size,:],
                y: data_set_y_train[index * batch_size:(index + 1) * batch_size,:],
                neurnum: i_n})


    #####################################
    # GET INITIAL TRAIN/VAL/TEST LOSSES #
    #####################################
    print '... initial loss values'
    # Initialize loss records
    epoch = 0
    all_train_scores=[]
    all_val_epochs = []
    all_test_epochs = []
    all_train_epochs=[]
    all_val_scores=[]
    all_test_scores=[]

    # Training loss
    training_losses=[]
    for i_train_movie in xrange(n_train_batches):
        mini_iter = trainInds[i_train_movie]
        minibatch_avg_cost = validate_model(mini_iter)
        training_losses = numpy.append(training_losses,minibatch_avg_cost)
    this_train_loss = numpy.sum(training_losses)
    all_train_scores = numpy.append(all_train_scores,this_train_loss)
    all_train_epochs = numpy.append(all_train_epochs,epoch)

    # Validation loss
    validation_losses=[]
    for i_val_movie in xrange(n_valid_batches):
        temp_val_cost = validate_model(valInds[i_val_movie])
        validation_losses = numpy.append(validation_losses,temp_val_cost)

    this_validation_loss = numpy.sum(validation_losses)
    all_val_scores = numpy.append(all_val_scores,this_validation_loss)
    all_val_epochs = numpy.append(all_val_epochs,epoch)

    # Test loss
    test_losses=[]
    test_losses, test_pred_temp, test_actual_temp = test_model()
    test_score = numpy.sum(test_losses)
    all_test_scores = numpy.append(all_test_scores,test_score)
    all_test_epochs = numpy.append(all_test_epochs,epoch)

    ###############
    # TRAIN MODEL #
    ###############

    print '... training'

    # early-stopping parameters
    patience = 3000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.99999999  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatchs before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()


    done_looping = False



    while (not done_looping): # Keep looping until converged according to early stopping criteria
        epoch = epoch + 1
        numpy.random.shuffle(trainInds) # Shuffle the order of train movies presented every epoch
        
        # Training section
        training_losses=[]
        for i_train_movie in xrange(n_train_batches):
            mini_iter = trainInds[i_train_movie]
            minibatch_avg_cost = train_model(mini_iter)
            training_losses = numpy.append(training_losses,minibatch_avg_cost)
            print minibatch_avg_cost
        this_train_loss = numpy.sum(training_losses)
        all_train_scores = numpy.append(all_train_scores,this_train_loss)
        all_train_epochs = numpy.append(all_train_epochs,epoch)

        # Iteration number
        iter = (epoch - 1) * n_train_batches + i_train_movie

        # Validation and testing every epoch
        if (iter + 1) % validation_frequency == 0:
            
            # Validation Data
            validation_losses=[]
            for i_val_movie in xrange(n_valid_batches):
                temp_val_cost = validate_model(valInds[i_val_movie])
                validation_losses = numpy.append(validation_losses,temp_val_cost)
        
            this_validation_loss = numpy.sum(validation_losses)
            all_val_scores = numpy.append(all_val_scores,this_validation_loss)
            all_val_epochs = numpy.append(all_val_epochs,epoch)
            
            print('epoch %i, minibatch %i, validation error %f' %
                 (epoch, i_train_movie+ 1,
                  this_validation_loss))

            # If best validaton score so far
            if this_validation_loss < best_validation_loss:
                
                # Improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                    patience = max(patience, iter * patience_increase)

                best_validation_loss = this_validation_loss
                best_iter = iter
                best_params = copy.deepcopy(params)

                # Test Data
                #test_losses=numpy.zeros((Ncells,))
                #test_pred=numpy.zeros((Ncells,3600))
                #test_actual=numpy.zeros((Ncells,3600))
                    #for i_test_movie in xrange(n_test_batches): #no more looping because I'm fitting a trial average version
                test_losses, test_pred, test_actual = test_model()
 
                test_score = numpy.sum(test_losses)
                all_test_scores = numpy.append(all_test_scores,test_score)
                all_test_epochs = numpy.append(all_test_epochs,epoch)
                
                print(('     epoch %i, minibatch %i, test error of '
                       'best model %f') %
                      (epoch, i_train_movie+ 1,
                       numpy.sum(test_score)))
                if math.isnan(test_score):
                    break
        if patience <= iter:
                done_looping = True
                break

        # Periodically save
        if numpy.any(numpy.equal(epoch,numpy.arange(1,5000,1))):
            f = file('RNN50_2layers/RNN50_2layers_'+exp+'_'+cell_type+'_parallelneurons_padzero_undone.save', 'wb')
            for obj in [[best_params] + [test_score] + [test_losses] + [test_pred] + [test_actual] + [all_train_epochs] + [all_train_scores]  + [all_val_epochs] + [all_val_scores] + [all_test_epochs] + [all_test_scores] + [epoch] + [valInds] + [trainInds] + [params]]:
                cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()


    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f'
           'obtained at iteration %i, with test performance %f') %
          (best_validation_loss, best_iter + 1, numpy.sum(test_score)))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
	
    # Save data
    f = file('RNN50_2layers/RNN50_2layers_'+exp+'_'+cell_type+'_parallelneurons_padzero.save', 'wb')
    for obj in [[best_params] + [test_score] + [test_losses] + [test_pred] + [test_actual] + [all_train_epochs] + [all_train_scores]  + [all_val_epochs] + [all_val_scores] + [all_test_epochs] + [all_test_scores] + [epoch] + [valInds] + [trainInds] + [params]]:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

#loop over neurons and cross validation parameters
if __name__ == '__main__':
    SGD_training()


