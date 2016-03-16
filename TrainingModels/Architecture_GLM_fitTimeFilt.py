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

import copy
######################
######################
# Create 2D filters  #
######################
######################

def matlab_style_gauss2D(shape=(3,3),sigma2=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    http://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = numpy.ogrid[-m:m+1,-n:n+1]
    h = T.exp( -(x*x + y*y) / (2.*sigma2) )
    #h[ h < numpy.finfo(h.dtype).eps*h.max() ] = 0
    #sumh = h.sum()
    #if sumh != 0:
    #    h /= sumh
    return h


def matlab_style_gauss2D_deriv(shape=(3,3),sigma2=0.5):
    """
    2D gaussian mask derivative
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = numpy.ogrid[-m:m+1,-n:n+1]
    h = numpy.exp( -(x*x + y*y) / (2.*sigma2) ) * (-2*sigma2 + x*x + y*y)/(4*numpy.pi*sigma2^3)
    return h
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

class LeNetConvPoolLayer_temporal(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), outputType = 'rl'):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        self.input = input
        fan_in = 20
        W_init = numpy.asarray(numpy.random.randn(1,20,1,1,1)*sqrt(2.0/fan_in),dtype=theano.config.floatX)
        self.W = theano.shared(value=W_init, name='W', borrow=True)
        self.W_helper = theano.shared(value=numpy.zeros((1,filter_shape[1],1,1,1), \
            dtype=theano.config.floatX), name='W_helper', borrow=True)
        self.W_helper2 = theano.shared(value=numpy.zeros((1,filter_shape[1],1,1,1), \
            dtype=theano.config.floatX), name='W_helper2', borrow=True)

        # parameters of this layer
        self.params = [self.W]
        self.params_helper = [self.W_helper]
        self.params_helper2 = [self.W_helper2]


        # to get same using 'valid', pre-pad with zeros
        image_shape_pad = list(image_shape)
        a1 = numpy.floor((filter_shape[1]-1)/2.0).astype(int)
        b1 = numpy.ceil((filter_shape[1]-1)/2.0).astype(int)
        #a2 = numpy.floor((filter_shape[3]-1)/2.0).astype(int)
        #b2 = numpy.ceil((filter_shape[3]-1)/2.0).astype(int)
        #a3 = numpy.floor((filter_shape[4]-1)/2.0).astype(int)
        #b3 = numpy.ceil((filter_shape[4]-1)/2.0).astype(int)

        image_shape_pad[1] += a1+b1
        #image_shape_pad[3] += a2+b2
        #image_shape_pad[4] += a3+b3

        input_padded = theano.shared(value=numpy.zeros(image_shape_pad, \
            dtype=theano.config.floatX), borrow=True)

        #input_padded = T.set_subtensor(input_padded[:,a1:-b1,:,a2:-b2,a3:-b3], input)
        input_padded = T.set_subtensor(input_padded[:,(a1+b1):,:,:,:], input)

        #post-pad
        #input_padded = T.concatenate( (input_padded,T.alloc(0,(1,b1,1,1,1))), axis = 1) #time
        #input_padded = T.concatenate( (input_padded,T.alloc(0,(1,1,1,b2,1))), axis = 3) #height
        #input_padded = T.concatenate( (input_padded,T.alloc(0,(1,1,1,1,b3))), axis = 4) #width

        conv_out = conv3d2d.conv3d(
            signals=input_padded,  # Ns, Ts, C, Hs, Ws
            filters=self.W, # Nf, Tf, C, Hf, Wf
            signals_shape=image_shape_pad, #(batchsize, in_time, in_channels, in_height, in_width)
            filters_shape=filter_shape, #(flt_channels, flt_time, in_channels, flt_height, flt_width)
            border_mode='valid')

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,ds=poolsize, ignore_border=True)

        self.lin_output = pooled_out;

        # Activation is given by sigmoid:
        #self.output = T.tanh(lin_output)

        # Activation is rectified linear
        if outputType == 'rl':
            self.output = self.lin_output*(self.lin_output>0)
        elif outputType == 'l':
            self.output = self.lin_output

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        an activation function (see below). Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer (overwritten in body of class)
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        # if W is None:
        W_values = numpy.asarray(rng.uniform(
                    low=-.2*numpy.sqrt(6. / (n_in + n_out)),
                    high=.2*numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
                  #if activation == theano.tensor.nnet.sigmoid:
                  #W_values *= 4
                  #W_values = numpy.random.randn(n_in, n_out).astype(theano.config.floatX)
       
       #W_unnorm = numpy.random.randn(n_in,n_out).astype(theano.config.floatX)
       # W_norms = numpy.sqrt(numpy.sum(W_unnorm**2,axis=0))
       #W = W_unnorm / W_norms
       #self.W = theano.shared(value=W,
       #                      name='W', borrow=True)
                               
        #W_values, s, v = numpy.linalg.svd(Q)
        #W_values = 1e-2*numpy.random.randn(n_in,n_out).astype(theano.config.floatX)
        W = theano.shared(value=W_values, name='W_hid', borrow=True)

    #if b is None:
        b_values = -2*numpy.ones((n_out,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, name='b_hid', borrow=True)

        self.W = W
        self.b = b

        # helper variables for adagrad
        self.W_helper = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_helper', borrow=True)
        self.b_helper = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_helper', borrow=True)

        # helper variables for L1
        self.W_helper2 = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_helper2', borrow=True)
        self.b_helper2 = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_helper2', borrow=True)


        # helper variables for lr
        self.W_helper3 = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_helper3', borrow=True)
        self.b_helper3 = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_helper3', borrow=True)
        
        # helper variables for stepsize
        self.W_helper4 = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_helper4', borrow=True)
        self.b_helper4 = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_helper4', borrow=True)
            
        # parameters of this layer
        self.params = [self.W, self.b]
        #self.params_helper = [self.W_helper]
        # self.params_helper2 = [self.W_helper2, self.b_helper2]
        #self.params_helper3 = [self.W_helper3, self.b_helper3]
    #self.params_helper4 = [self.W_helper4, self.b_helper4]

        lin_output = T.dot(input, self.W) + self.b

        # Hidden unit activation is given by: tanh(dot(input,W) + b)
        #self.output = T.tanh(lin_output)

        # Hidden unit activation is rectified linear
        self.output = lin_output #*(lin_output>0)

        # Hidden unit activation is None (i.e. linear)
        #self.output = lin_output


class PoissonRegression(object):
    """Poisson Regression Class

    The poisson regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. 
    """

    def __init__(self, input, n_in, n_out, obj_function):
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
      
        # compute vector of expected values (for each output) in symbolic form
        if obj_function == 'poiss':
            self.E_y_given_x = T.exp(input)
        elif obj_function == 'gauss':
            self.E_y_given_x = T.dot(input,self.W) + self.b
        elif obj_function == 'logexp':
            self.E_y_given_x = T.log(1 + T.exp(T.dot(input,self.W) + self.b))

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
        if self.obj_f_type == 'poiss':
            return -T.sum((  (y * T.log(self.E_y_given_x)) - (self.E_y_given_x)  ) , axis = 0)
        elif self.obj_f_type == 'gauss':
            return T.sum((  (y  - self.E_y_given_x)**2) , axis = 0)
        elif self.obj_f_type == 'logexp':
            return -T.sum((  (y * T.log(self.E_y_given_x)) - (self.E_y_given_x)  ) , axis = 0)

        #return -T.sum( T.addbroadcast(maskData,1) * (  (y  - trialCount*self.E_y_given_x)**2) , axis = 0)
        #return -T.sum( maskData *(T.log( (self.E_y_given_x.T ** y) * T.exp(-self.E_y_given_x.T) / T.gamma(y+1) )) )

    def errors(self, y, trialCount, maskData):
        """Use summed absolute value of difference between actual number of spikes per bin and predicted E[y|x]

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        return T.sum(  maskData * T.sqrt(((trialCount*self.E_y_given_x)-y) ** 2)  )

######################
######################
# LOAD DATA FUNCTION #
######################
######################

"""
Loads in the prepared dataset
stimuli are images and responses are neural activity of parsol cells

"""

def load_data(i):
    ''' Loads the dataset
    '''

    #############
    # LOAD DATA #
    #############
    print '... loading data'
    dataset='raw_NSEM.mat'
    f = h5py.File(dataset)
    stimuli_train = numpy.array(f['FitMovies'])/255.
    responses_train2 = numpy.array(f['FitResponses'])
    stimuli_test = numpy.array(f['TestMovie'])/255.
    responses_test2 = numpy.array(f['TestResponses'])
    trainBatchSize = numpy.array(f['fit_batch'])
    testBatchSize = numpy.array(f['test_batch'])
    RGC_locations = numpy.array(f['locations'])
    temporalFilter = numpy.array(f['tempFilt'])
    f.close()

#    stimuli_test2 = numpy.concatenate([stimuli_test3,numpy.zeros((stimuli_test3.shape[0],80))],axis=1)
#    stimuli_train2 = numpy.concatenate([stimuli_train3,numpy.zeros((stimuli_train3.shape[0],80))],axis=1)


    responses_train = responses_train2[:,i]
    responses_test= responses_test2[:,i]
    
    ## Cut out image patch
    #    ordered_rgc_indices2 = numpy.zeros((1,31*31))
    #image_patch_size = [30, 30]
    #yinds = [RGC_locations[0,i].astype(int)-image_patch_size[0]/2,RGC_locations[0,i].astype(int)+image_patch_size[0]/2+1]
    #xinds = [RGC_locations[1,i].astype(int)-image_patch_size[1]/2,RGC_locations[1,i].astype(int)+image_patch_size[1]/2+1]
    #flatInds = numpy.arange(0,3200,1).reshape(40,80)
    #flatInds = numpy.concatenate([flatInds,3201*numpy.ones((20,80))])
    #flatInds = numpy.concatenate([flatInds,3201*numpy.ones((60,20))],axis=1)
    #theseInds = flatInds[yinds[0]-1:yinds[1]-1,xinds[0]-1:xinds[1]-1]
    #hh = theseInds.flatten()
    #ordered_rgc_indices2[0,:] = hh

    #stimuli_train = stimuli_train2[:, ordered_rgc_indices2[0,:].astype(int64)]


    ## Cut out image patch
    #   stimuli_test =  stimuli_test2[:, ordered_rgc_indices2[0,:].astype(int64)]
    
    
    #subset of data
    #stimuli_train = stimuli_train[1:1e5,:]
    #responses_train = responses_train[1:1e5,:]
    #stimuli_test = stimuli_test[1:1e5,:]
    #responses_test = responses_test[1:1e5,:]

    #off parasols only
    #responses_train = responses_train[:,0:4]
    #responses_test = responses_test[:,0:4]
    #RGC_locations = RGC_locations[:,0:4]

    #on parasols only
    #responses_train = responses_train[:,4:]
    #responses_test = responses_test[:,4:]
    #RGC_locations = RGC_locations[:,4:]

    #temporal offset?
    #responses = responses[1:-1][:]
    #stimuli = stimuli[0:-2][:]

    print '... done loading'

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
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue

        #return shared_x, T.cast(shared_y, 'int32')	
        return shared_x, shared_y

    data_set_x_train, data_set_y_train = shared_dataset(data_set_train)
    data_set_x_test, data_set_y_test = shared_dataset(data_set_test)
    data_set_x_test = T.concatenate([data_set_x_test,T.zeros((T.shape(data_set_x_test)[0],80))],axis=1)
    data_set_x_train = T.concatenate([data_set_x_train,T.zeros((T.shape(data_set_x_train)[0],80))],axis=1)
    rval = [(data_set_x_train, data_set_y_train), (data_set_x_test, data_set_y_test), trainBatchSize, testBatchSize, RGC_locations, temporalFilter]

    return rval


######################
######################
# build and train    #
######################
######################

def SGD_training(learning_rate=5e-3, L1_reg=0, L2_reg=0, n_epochs=1000):
    """
    stochastic gradient descent optimization for a multilayer
    perceptron

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

   """
    n_neur= int(sys.argv[1])
    
    dataset_info = load_data(i=n_neur)

    data_set_x_train, data_set_y_train = dataset_info[0]
    data_set_x_test, data_set_y_test = dataset_info[1]
    trainBatchSize = int(dataset_info[2][0][0])
    testBatchSize = int(dataset_info[3][0][0])
    batch_size = min(trainBatchSize,testBatchSize) #train and test batch sizes should be same size
    RGC_locations = dataset_info[4]
    temporalFilter = dataset_info[5]

    # compute number of minibatches for training, validation and testing
    n_train_batches = (data_set_y_train.get_value(borrow=True).shape[0])/batch_size
    n_valid_batches = 10
    n_test_batches = (data_set_y_test.get_value(borrow=True).shape[0])/testBatchSize - n_valid_batches

    ordered_rgc_indices2 = numpy.zeros((9,31*31))
    i=n_neur
    image_patch_size = [30, 30]
    yinds = [RGC_locations[0,i].astype(int)-image_patch_size[0]/2,RGC_locations[0,i].astype(int)+image_patch_size[0]/2+1]
    xinds = [RGC_locations[1,i].astype(int)-image_patch_size[1]/2,RGC_locations[1,i].astype(int)+image_patch_size[1]/2+1]
    flatInds = numpy.arange(0,3200,1).reshape(40,80)
    flatInds = numpy.concatenate([flatInds,3201*numpy.ones((20,80))])
    flatInds = numpy.concatenate([flatInds,3201*numpy.ones((60,20))],axis=1)
    theseInds = flatInds[yinds[0]-1:yinds[1]-1,xinds[0]-1:xinds[1]-1]
    hh = theseInds.flatten()
    #hh = numpy.concatenate((yinds,xinds),axis=0) -1
    ordered_rgc_indices2[0,:] = hh
    
    ordered_rgc_indices= theano.shared(numpy.asarray(ordered_rgc_indices2,dtype=int64),borrow=True)
    

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as a vector of inputs with many exchangeable examples of this vector
    y = T.vector('y')  # the output is a vector of matched output unit responses.

    rng = numpy.random.RandomState(1234)

    #####################################################################################
    # Architecture: input --> temporal filtering --> some intermediate layers --> Poisson observations
    #####################################################################################
    nkerns= [1]
    fside = [20]
    #subunit_var_init = [2, 2]
    
    reshaped_input = x.reshape((batch_size, 1, 31, 31))
    
    reshaped_input = T.shape_padleft(  x.reshape((batch_size, 1, 31, 31))  , n_ones=1)

    Layer1 = LeNetConvPoolLayer_temporal(rng, input = reshaped_input, filter_shape=(1, len(temporalFilter), 1, 1, 1), image_shape = (1, batch_size, 1, 31, 31), poolsize=(1, 1), outputType = 'l')

    Layer2 = HiddenLayer(rng,input=Layer1.output.reshape((batch_size, 31*31)),n_in=31*31,n_out=1)
    Layer2b_input = Layer2.output.reshape((batch_size,))
    Layer2b = PoissonRegression(input=Layer2b_input,n_in=1,n_out=1,obj_function='poiss')

    #######################
    # Objective function
    #######################

    # L1 norm ; one regularization option is to enforce L1 norm to
    # be small
    L1 = abs(Layer2.W).sum() # + abs(Layer1.W).sum()

    # square of L2 norm ; one regularization option is to enforce
    # square of L2 norm to be small
    L2_sqr = (Layer2.W ** 2).sum() # + (Layer1.W ** 2).sum()

    negative_log_likelihood = Layer2b.negative_log_likelihood

    errors = Layer2b.errors

    # create a list (concatenated) of all model parameters to be fit by gradient descent

    ################################################
    # Architecture params
    ################################################
    #order: [self.W, self.b, self.h] + [self.W, self.b, self.S] + [self.sigma2, self.b]
    params = Layer2.params + Layer1.params
    #params_helper = Layer2b.params_helper + Layer2.params_helper #+ Layer1.params_helper
    #params_helper2 = Layer2b.params_helper2 + Layer2.params_helper2 #+ Layer1.params_helper2

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = T.sum(negative_log_likelihood(y)) \
         + L1_reg * L1 \
         + L2_reg * L2_sqr

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    # use cost or errors(y,tc,md) as output?
    test_model = theano.function(inputs=[index],
            #outputs=[negative_log_likelihood(y), Layer2b.E_y_given_x, y, Layer1.lin_output,Layer2.lin_output],
            outputs=[negative_log_likelihood(y), Layer2b.E_y_given_x, y],
            givens={
                x: data_set_x_test[0 * testBatchSize:(0 + 1) * batch_size, ordered_rgc_indices[0,:]],
                y: data_set_y_test[index * testBatchSize:(index + 1) * batch_size]}) #by default, indexes first dimension which is samples

    # wanted to use below indexes and have different sized batches, but this didn't work
    #[int(batchBreaks[index]-1):int(batchBreaks[(index+1)]-1)]

    validate_model = theano.function(inputs=[index],
            outputs=T.sum(negative_log_likelihood(y)),
            givens={
                x: data_set_x_test[0 * testBatchSize:(0 + 1) * batch_size, ordered_rgc_indices[0,:]],
                y: data_set_y_test[index * testBatchSize:(index + 1) * batch_size]})


    updates = Adam(cost, params)

    """
    The next bit of code forms the updates.  ADAM might be a good choice to replace this section (e.g. https://gist.github.com/Newmu/acb738767acb4788bac3). At various points we wanted to explore different updates so they are all coded here.  In some cases we wanted projected updates to enforce the parameter to remain positive or negative. 

    """


    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index], outputs=cost,
            updates=updates,
            givens={
                x: data_set_x_train[index * batch_size:(index + 1) * batch_size, ordered_rgc_indices[0,:]],
                y: data_set_y_train[index * batch_size:(index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 3000  # look as this many examples regardless
    #patience = train_set_x.get_value(borrow=True).shape[0] * n_epochs #no early stopping
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.99999999  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()
    all_val_epochs = []
    all_test_epochs = []
    all_val_scores=[]
    all_test_scores=[]
    epoch = 0
    done_looping = False
    loop_vec = numpy.arange(n_train_batches)
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        numpy.random.shuffle(loop_vec)
        for minibatch_index in xrange(n_train_batches):
            mini_iter = loop_vec[minibatch_index]
            minibatch_avg_cost = train_model(mini_iter)
            print minibatch_avg_cost
            if math.isnan(minibatch_avg_cost):
                break
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute absolute error loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                all_val_scores = numpy.append(all_val_scores,this_validation_loss)
                all_val_epochs = numpy.append(all_val_epochs,epoch)
                print('epoch %i, minibatch %i, validation error %f' %
                     (epoch, minibatch_index + 1,
                      this_validation_loss))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    best_params = copy.deepcopy(params)
                    for i in xrange(n_valid_batches,n_valid_batches+n_test_batches):
                        test_losses_i, test_pred_i, test_actual_i = test_model(i)
                        if i==n_valid_batches:
                            test_losses = test_losses_i
                            test_pred = test_pred_i
                            test_actual = test_actual_i
                        # input_responses = input_responses_i
                        else:
                            test_losses += test_losses_i
                            test_pred += test_pred_i
                            test_actual += test_actual_i

                    test_score = numpy.mean(test_losses)
                    all_test_scores = numpy.append(all_test_scores,test_score)
                    all_test_epochs = numpy.append(all_test_epochs,epoch)
                    print(('     epoch %i, minibatch %i, test error of '
                           'best model %f') %
                          (epoch, minibatch_index + 1,
                           numpy.sum(test_score)))
                    if math.isnan(test_score):
                        break
            if patience <= iter:
                    done_looping = True
                    break
            if numpy.any(numpy.equal(epoch,numpy.arange(5,5000,5))):
                  f = file('nat_parasols_sigUpdate' + '_lam_' + str(L1_reg) + '_nk_' + str(nkerns[0]) + '_GLM_N' + str(n_neur) + '_fitTempFilt _undone.save', 'wb')
                  for obj in [best_params + [test_score] + [test_losses] + [test_pred] + [test_actual]+ [all_val_epochs] + [all_val_scores] + [all_test_epochs] + [all_test_scores] + [epoch]]:
                        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
                  f.close()

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f'
           'obtained at iteration %i, with test performance %f') %
          (best_validation_loss, best_iter + 1, numpy.sum(test_score)))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
	
    #store data
    f = file('nat_parasols_sigUpdate' + '_lam_' + str(L1_reg) + '_nk_' + str(nkerns[0]) + '_GLM_N' + str(n_neur) + '_fitTempFilt.save', 'wb')
    for obj in [best_params + [test_score] + [test_losses] + [test_pred] + [test_actual]+ [all_val_epochs] + [all_val_scores] + [all_test_epochs] + [all_test_scores] + [epoch]]:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

#loop over neurons and cross validation parameters
if __name__ == '__main__':
    SGD_training(L1_reg=0,L2_reg=0)


