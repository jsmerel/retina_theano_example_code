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

import numpy
import h5py
from pylab import *

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.tensor.nnet import conv3d2d

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
# LOAD DATA FUNCTION #
######################
######################

"""
Loads in the prepared dataset
stimuli are images and responses are neural activity of parsol cells

"""

def load_data():
    ''' Loads the dataset
    '''

    #############
    # LOAD DATA #
    #############
    print '... loading data'
    dataset='../raw_NSEM.mat'
    f = h5py.File(dataset)
    stimuli_train = numpy.array(f['FitMovies']) 
    responses_train = numpy.array(f['FitResponses'])
    stimuli_test = numpy.array(f['TestMovie']) 
    responses_test = numpy.array(f['TestResponses'])
    trainBatchSize = numpy.array(f['fit_batch'])
    testBatchSize = numpy.array(f['test_batch'])
    RGC_locations = numpy.array(f['locations'])
    temporalFilter = numpy.array(f['tempFilt'])
    f.close()

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
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
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

    rval = [(data_set_x_train, data_set_y_train), (data_set_x_test, data_set_y_test), trainBatchSize, testBatchSize, RGC_locations, temporalFilter]

    return rval


######################
######################
# Conv Layer Class   #
######################
######################

#This class is to build the LeNet-style convolution + max pooling layers + output nonlinearity
class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), outputType = 'rl', EI_layer = False, filter_init = None):
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

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(1. / (fan_in + fan_out))/1e3
        if filter_init is None:
            W_init = numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=theano.config.floatX)
        else:
            W_init = filter_init.astype(theano.config.floatX)

        self.W = theano.shared(W_init, borrow=True)
        #self.W = theano.shared(value=numpy.zeros(filter_shape, dtype=theano.config.floatX), borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # helper variables for adagrad
        self.W_helper = theano.shared(value=numpy.zeros(filter_shape, \
            dtype=theano.config.floatX), name='W_helper', borrow=True)
        self.b_helper = theano.shared(value=numpy.zeros((filter_shape[0],), \
            dtype=theano.config.floatX), name='b_helper', borrow=True)

        # helper variables for L1
        self.W_helper2 = theano.shared(value=numpy.zeros(filter_shape, \
            dtype=theano.config.floatX), name='W_helper2', borrow=True)
        self.b_helper2 = theano.shared(value=numpy.zeros((filter_shape[0],), \
            dtype=theano.config.floatX), name='b_helper2', borrow=True)

        # parameters of this layer
        self.params = [self.W, self.b]
        self.params_helper = [self.W_helper, self.b_helper]
        self.params_helper2 = [self.W_helper2, self.b_helper2]

        # convolve input feature maps with filters
        if EI_layer:
            conv_out = conv.conv2d(input=input, filters=T.set_subtensor(self.W[:,0::2,:,:], -self.W[:,0::2,:,:]),
                filter_shape=filter_shape, image_shape=image_shape, border_mode='full')
        else:
            conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape, border_mode='full')

        #convert to "same"
        s1 = numpy.floor((filter_shape[2]-1)/2.0).astype(int)
        e1 = numpy.ceil((filter_shape[2]-1)/2.0).astype(int)
        s2 = numpy.floor((filter_shape[3]-1)/2.0).astype(int)
        e2 = numpy.ceil((filter_shape[3]-1)/2.0).astype(int)
        conv_out = conv_out[:,:,s1:-e1,s2:-e2]
                
        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height

        self.lin_output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x');

        # Activation is given by sigmoid:
        #self.output = T.tanh(lin_output)

        # Activation is rectified linear
        if outputType == 'rl':
            self.output = self.lin_output*(self.lin_output>0)
        elif outputType == 'l':
            self.output = self.lin_output

#######################
#######################
# Conv Layer Class    #
#(with full component)#
#######################
#######################

"""
This convolutional layer functions the same as the standard layer but there is an additional "full component", S, that is pointwise multiplied by the input and added to the convolutional response maps.

"""

#This class is to build the LeNet-style convolution + max pooling layers + output nonlinearity
class LeNetConvPoolLayer_fullComponent(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), outputType = 'rl', EI_layer = False, filter_init = None):
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

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(1. / (fan_in + fan_out))
        if filter_init is None:
            W_init = numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=theano.config.floatX)
        else:
            W_init = filter_init.astype(theano.config.floatX)

        self.W = theano.shared(W_init, borrow=True)
        #self.W = theano.shared(value=numpy.zeros(filter_shape, dtype=theano.config.floatX), borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # helper variables for adagrad
        self.W_helper = theano.shared(value=numpy.zeros(filter_shape, \
            dtype=theano.config.floatX), name='W_helper', borrow=True)
        self.b_helper = theano.shared(value=numpy.zeros((filter_shape[0],), \
            dtype=theano.config.floatX), name='b_helper', borrow=True)

        # helper variables for L1
        self.W_helper2 = theano.shared(value=numpy.zeros(filter_shape, \
            dtype=theano.config.floatX), name='W_helper2', borrow=True)
        self.b_helper2 = theano.shared(value=numpy.zeros((filter_shape[0],), \
            dtype=theano.config.floatX), name='b_helper2', borrow=True)

        #include full component
        self.S = theano.shared(value=numpy.zeros((1,image_shape[1],image_shape[2],image_shape[3]), \
            dtype=theano.config.floatX), name='S', borrow=True, broadcastable=(True, False, False, False))
        self.S_helper = theano.shared(value=numpy.zeros((1,image_shape[1],image_shape[2],image_shape[3]), \
            dtype=theano.config.floatX), name='S_helper', borrow=True)
        self.S_helper2 = theano.shared(value=numpy.zeros((1,image_shape[1],image_shape[2],image_shape[3]), \
            dtype=theano.config.floatX), name='S_helper2', borrow=True)

        # parameters of this layer
        self.params = [self.W, self.b, self.S]
        self.params_helper = [self.W_helper, self.b_helper, self.S_helper]
        self.params_helper2 = [self.W_helper2, self.b_helper2, self.S_helper2]

        # convolve input feature maps with filters (it flips the kernel)
        if EI_layer:
            conv_out = conv.conv2d(input=input, filters=T.set_subtensor(self.W[:,0::2,:,:], -self.W[:,0::2,:,:]),
                filter_shape=filter_shape, image_shape=image_shape, border_mode='full')
        else:
            conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape, border_mode='full')

        #apply extra full component
        full_out = conv.conv2d(input=input*self.S, filters=numpy.ones(filter_shape),
                filter_shape=filter_shape, image_shape=image_shape, border_mode='full')

        #convert to "same"
        s1 = numpy.floor((filter_shape[2]-1)/2.0).astype(int)
        e1 = numpy.ceil((filter_shape[2]-1)/2.0).astype(int)
        s2 = numpy.floor((filter_shape[3]-1)/2.0).astype(int)
        e2 = numpy.ceil((filter_shape[3]-1)/2.0).astype(int)
        conv_out = conv_out[:,:,s1:-e1,s2:-e2]
        full_out = full_out[:,:,s1:-e1,s2:-e2]

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out+full_out,ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height

        self.lin_output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x');

        # Activation is given by sigmoid:
        #self.output = T.tanh(lin_output)

        # Activation is rectified linear
        if outputType == 'rl':
            self.output = self.lin_output*(self.lin_output>0)
        elif outputType == 'l':
            self.output = self.lin_output


######################
######################
# Conv Layer Class   #
# (fixed filters)    #
######################
######################

"""
This class is a convolutional layer where the filters are Gaussians with some initial width (variance).  The variance of the Gaussian is a parameter and can be updated via the SGD optimization if desired.

"""

#This class is to build the LeNet-style convolution + max pooling layers + output nonlinearity
class LeNetConvPoolLayer_fixed(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, filter_var_init, image_shape, poolsize=(2, 2), outputType = 'rl'):
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

        assert image_shape[1] == filter_shape[1]
        self.input = input

        self.sigma2 = theano.shared(numpy.asarray(filter_var_init).astype(theano.config.floatX), borrow=True)
 
        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # helper variables for adagrad
        self.sigma2_helper = theano.shared(value=numpy.zeros(len(filter_var_init), \
            dtype=theano.config.floatX), name='sigma2_helper', borrow=True)
        self.b_helper = theano.shared(value=numpy.zeros((filter_shape[0],), \
            dtype=theano.config.floatX), name='b_helper', borrow=True)

        # helper variables for L1
        self.sigma2_helper2 = theano.shared(value=numpy.zeros(len(filter_var_init), \
            dtype=theano.config.floatX), name='sigma2_helper2', borrow=True)
        self.b_helper2 = theano.shared(value=numpy.zeros((filter_shape[0],), \
            dtype=theano.config.floatX), name='b_helper2', borrow=True)

        self.W = T.concatenate((-.1*matlab_style_gauss2D((filter_shape[2],filter_shape[3]),self.sigma2[0]).reshape((1, filter_shape[1], filter_shape[2], filter_shape[3])), .1*matlab_style_gauss2D((filter_shape[2],filter_shape[3]),self.sigma2[1]).reshape((1, filter_shape[1], filter_shape[2], filter_shape[3]))), axis=0) 

        # parameters of this layer
        self.params = [self.sigma2, self.b]
        self.params_helper = [self.sigma2_helper, self.b_helper]
        self.params_helper2 = [self.sigma2_helper2, self.b_helper2]

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
            filter_shape=filter_shape, image_shape=image_shape, border_mode='full')

        #convert to "same"
        s1 = numpy.floor((filter_shape[2]-1)/2.0).astype(int)
        e1 = numpy.ceil((filter_shape[2]-1)/2.0).astype(int)
        s2 = numpy.floor((filter_shape[3]-1)/2.0).astype(int)
        e2 = numpy.ceil((filter_shape[3]-1)/2.0).astype(int)
        conv_out = conv_out[:,:,s1:-e1,s2:-e2]

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height

        self.lin_output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x');

        # Activation is given by sigmoid:
        #self.output = T.tanh(lin_output)

        # Activation is rectified linear
        if outputType == 'rl':
            self.output = self.lin_output*(self.lin_output>0)
        elif outputType == 'l':
            self.output = self.lin_output

    def getW(self,filter_shape):
        W=numpy.zeros(filter_shape)
        W[0,:,:,:] = -.1*matlab_style_gauss2D((filter_shape[2],filter_shape[3]),self.sigma2[0])/numpy.max(matlab_style_gauss2D((filter_shape[2],filter_shape[3]),self.sigma2[0]))
        W[1,:,:,:] = .1*matlab_style_gauss2D((filter_shape[2],filter_shape[3]),self.sigma2[1])/numpy.max(matlab_style_gauss2D((filter_shape[2],filter_shape[3]),self.sigma2[1]))
        return W

    def getW_deriv(self):
        return 1



######################
######################
# Conv Layer Class   #
# (temporal filters) #
######################
######################

"""
This is a convolution with an extra dimension for time.  I've only been using with temporal filtering only.  Effectively, if it is the first layer in the architecture, I'm temporally preprocessing the input before running spatial filters over it.  

"""

#This class is to build the LeNet-style convolution + max pooling layers + output nonlinearity
class LeNetConvPoolLayer_temporal(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, temporal_filter, image_shape, poolsize=(2, 2), outputType = 'rl'):
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

        self.W = theano.shared(value=numpy.reshape(temporal_filter,(1,filter_shape[1],1,1,1)).astype(theano.config.floatX), name='W', borrow=True)
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




################################
################################
# Diagonal Poisson Layer Class #
# (still has biases)           #
################################
################################

"""
A Layer with a poisson likelihood (essentially a poisson GLM layer which would go from hidden units to observed neurons)

"""

#This variant is the standard multiple input, multiple output version
class PoissonRegressionD(object):
    """Poisson Regression Class

    The poisson regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. 
    """

    def __init__(self, input, n_filt, n_in, n_out, y, hist_len, y_len):
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
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        #self.W = theano.shared(value=numpy.identity(n_in, dtype=theano.config.floatX), name='W', borrow=True)
        self.W = theano.shared(value=numpy.tile([-1, -1, -1, -1, 1, 1, 1, 1, 1]*numpy.ones((n_in,), dtype=theano.config.floatX)/n_filt, (1,n_filt,1) ).astype(theano.config.floatX), name='W', borrow=True)
        #self.W = theano.shared(value=numpy.concatenate((-1*numpy.ones((4,)),numpy.ones((5,)),-1*numpy.ones((4,)),numpy.ones((5,))))*numpy.ones((n_in,), dtype=theano.config.floatX), name='W', borrow=True)
        #self.W = theano.shared(value=.001*numpy.ones((n_in,), dtype=theano.config.floatX), name='W', borrow=True)

        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=-1*numpy.ones((n_out,), dtype=theano.config.floatX), name='b', borrow=True)

        self.h = theano.shared(value=numpy.zeros((hist_len,n_out), dtype=theano.config.floatX), name='h', borrow=True)

        # helper variables for adagrad
        self.b_helper = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_helper', borrow=True)
        self.W_helper = theano.shared(value=numpy.tile(numpy.zeros((n_in,), \
            dtype=theano.config.floatX), (1,n_filt,1) ), name='W_helper', borrow=True)
        self.h_helper = theano.shared(value=numpy.zeros((hist_len,n_out), dtype=theano.config.floatX), name='h_helper', borrow=True)

        # helper variables for L1
        self.b_helper2 = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_helper2', borrow=True)
        self.W_helper2 = theano.shared(value=numpy.tile(numpy.zeros((n_in,), \
            dtype=theano.config.floatX), (1,n_filt,1) ), name='W_helper2', borrow=True)
        self.h_helper2 = theano.shared(value=numpy.zeros((hist_len,n_out), dtype=theano.config.floatX), name='h_helper', borrow=True)

        # parameters of the model
        self.params = [self.W, self.b, self.h]
        self.params_helper = [self.W_helper, self.b_helper, self.h_helper]
        self.params_helper2 = [self.W_helper2, self.b_helper2, self.h_helper2]

        #history dependent input
        self.h_in = theano.shared(value=numpy.zeros((y_len,n_out), dtype=theano.config.floatX), borrow=True) 
        for hi in xrange(hist_len):
            self.h_in = T.set_subtensor(self.h_in[(1+hi):y_len], self.h_in[(1+hi):y_len] + T.addbroadcast(T.shape_padleft(self.h[hi,:],n_ones=1),0)*y[0:(y_len-(hi+1))])

        # compute vector of expected values (for each output) in symbolic form
        self.E_y_given_x = T.log(1+T.exp(T.sum(input*T.addbroadcast(self.W,0), axis=1) + self.b + self.h_in)) #sums over multiple filters
        self.input_responses = T.sum(input*T.addbroadcast(self.W,0), axis=1) + self.b #sums over multiple filters
        #self.E_y_given_x = T.exp(T.dot(input, self.W) + self.b) #possible alternative

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
        return -T.sum( (  (y * T.log(self.E_y_given_x)) - (self.E_y_given_x)  ) , axis = 0)
        #return -T.sum( maskData *(T.log( (self.E_y_given_x.T ** y) * T.exp(-self.E_y_given_x.T) / T.gamma(y+1) )) )

    def errors(self, y):
        """Use summed absolute value of difference between actual number of spikes per bin and predicted E[y|x]

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        return T.sum( T.sqrt(((self.E_y_given_x)-y) ** 2)  )





######################
######################
# build and train    #
######################
######################

def SGD_training(learning_rate=1e-4, L1_reg=0, L2_reg=0, n_epochs=1000):
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
    dataset_info = load_data()

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


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as a vector of inputs with many exchangeable examples of this vector
    y = T.matrix('y')  # the output is a vector of matched output unit responses.

    rng = numpy.random.RandomState(1234)

    #####################################################################################
    # Architecture: input --> temporal filtering --> some intermediate layers --> Poisson observations
    #####################################################################################
    nkerns= [1]
    fside = [20]
    #subunit_var_init = [2, 2]

    # Reshape matrix of rasterized images of shape (batch_size,52*50)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    reshaped_input = x.reshape((batch_size, 1, 40, 80))

    reshaped_input = T.shape_padleft(  x.reshape((batch_size, 1, 40, 80))  , n_ones=1) 

    # first do temporal filtering
    Layer0 = LeNetConvPoolLayer_temporal(rng, input = reshaped_input, filter_shape=(1, len(temporalFilter), 1, 1, 1), temporal_filter = temporalFilter, image_shape = (1, batch_size, 1, 40, 80), poolsize=(1, 1), outputType = 'l')
    
    # Two commented out layers: these would be example uses of the fixed subunits and the full component second layer (adjust "fside", which is the filter side length and nkerns above -- they need more than one element if there is more than one layer between temporal filtering and poisson observations)
    #
    # Construct the first convolutional pooling layer:
    # filtering doesn't reduce image if full is used
    # 4D output tensor is thus of shape (batch_size,nkerns[0],26,1)
    #Layer1 = LeNetConvPoolLayer_fixed(rng, input=Layer0.output.reshape((batch_size, 1, 40, 80)), filter_shape=(nkerns[0], 1, fside[0], fside[0]),
    #        filter_var_init = subunit_var_init, image_shape=(batch_size, 1, 40, 80), poolsize=(1, 1))

    # Construct the second convolutional pooling layer
    # filtering doesn't reduce image if full is used
    # maxpooling reduces this further to (25/2,26/2) = (12,13)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],12,13)
    #Layer2 = LeNetConvPoolLayer_fullComponent(rng, input=Layer1.output,
    #        image_shape=(batch_size, nkerns[0], 40, 80),
    #        filter_shape=(nkerns[1], nkerns[0], fside[1], fside[1]), poolsize=(1, 1), outputType = 'l', EI_layer = False)

    # going straight from temporal filtering to this layer is going to produce a GLM-like fit with one filter that all of the neurons share
    Layer2 = LeNetConvPoolLayer(rng, input=Layer0.output.reshape((batch_size, 1, 40, 80)),
            image_shape=(batch_size, 1, 40, 80),
            filter_shape=(nkerns[0], 1, fside[0], fside[0]), poolsize=(1, 1), outputType = 'l')

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # Flatten will generate a matrix of shape (batch_size,nkerns[0]*26*1)
    Layer2b_input = Layer2.output.flatten(3) #(batch_size, nkerns[0], 40*80)

    # select locations on convolutional map which correspond to actual RGC locations
    ordered_rgc_indices = numpy.ravel_multi_index(RGC_locations.astype(int)-1, dims=(40,80))
    Layer2b_input = Layer2b_input[:,:,ordered_rgc_indices.astype(int)] #(batch_size, nkerns[0], #neurons)

    # The poisson regression layer gets as input the hidden units
    # of the hidden layer (identity poisson doesn't reweight things)
    Layer2b = PoissonRegressionD(input=Layer2b_input, n_filt = nkerns[0], n_in=ordered_rgc_indices.size, n_out=ordered_rgc_indices.size, y=y, hist_len=10, y_len = batch_size)

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
    params = Layer2b.params + Layer2.params #+ Layer1.params
    params_helper = Layer2b.params_helper + Layer2.params_helper #+ Layer1.params_helper
    params_helper2 = Layer2b.params_helper2 + Layer2.params_helper2 #+ Layer1.params_helper2


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
            outputs=[negative_log_likelihood(y), Layer2b.input_responses, Layer2b.E_y_given_x, y],
            givens={
                x: data_set_x_test[0 * testBatchSize:(0 + 1) * batch_size], 
                y: data_set_y_test[index * testBatchSize:(index + 1) * batch_size]}) #by default, indexes first dimension which is samples

    # wanted to use below indexes and have different sized batches, but this didn't work
    #[int(batchBreaks[index]-1):int(batchBreaks[(index+1)]-1)]

    validate_model = theano.function(inputs=[index],
            outputs=numpy.sum(negative_log_likelihood(y)),
            givens={
                x: data_set_x_test[0 * testBatchSize:(0 + 1) * batch_size],
                y: data_set_y_test[index * testBatchSize:(index + 1) * batch_size]})

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = []
    for param in params:
        #gparam = theano.map(lambda yi,tci,mdi: T.grad(cost(yi,tci,mdi), param), sequences=[y,tc,md])
        gparam = T.grad(cost, param,disconnected_inputs='warn')
        gparams.append(gparam)


    """
    The next bit of code forms the updates.  ADAM might be a good choice to replace this section (e.g. https://gist.github.com/Newmu/acb738767acb4788bac3). At various points we wanted to explore different updates so they are all coded here.  In some cases we wanted projected updates to enforce the parameter to remain positive or negative. 

    """

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = []
    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    #for param, gparam in zip(params, gparams):
    #    updates.append((param, param - learning_rate * gparam))
    # adagrad
    iter_count = theano.shared(1)
    L1_penalized = []
    smaller_stepsize = [0,3]
    zero_stepsize = []
    #enforce_positive = [2, 3] #if recurrent
    enforce_positive = [] # [2,4] # is lower layer weights
    enforce_negative = [2]
    param_index = 0
    rho = 1e-6
    for param, param_helper, param_helper2, gparam in zip(params, params_helper, params_helper2, gparams):
        updates.append((param_helper, param_helper + gparam ** 2)) #need sum of squares for learning rate
        updates.append((param_helper2, param_helper2 + gparam)) #need sum of gradients for L1 thresholding
        if param_index in L1_penalized:
            updates.append( ( param, T.addbroadcast(T.maximum(0,T.sgn(T.abs_(param_helper2)/iter_count - L1_reg)) * (-T.sgn(param_helper2)*learning_rate*iter_count/(rho + (param_helper + gparam ** 2) ** 0.5) * (T.abs_(param_helper2)/iter_count - L1_reg)),0) ) )
        elif param_index in smaller_stepsize:
            updates.append((param, param - 1e-4*learning_rate * gparam)) #no adagrad
            #updates.append((param, param - learning_rate*1e2 * gparam / (rho + (param_helper + gparam ** 2) ** 0.5))) #adagrad
        elif param_index in zero_stepsize:
            pass
        elif param_index in enforce_positive:
            #updates.append((param, T.maximum(0, param - learning_rate * gparam / (rho + (param_helper + gparam ** 2) ** 0.5) )  )) #adagrad
            updates.append((param, T.maximum(0,param - 1*learning_rate * gparam))) #no adagrad
        elif param_index in enforce_negative:
            #updates.append((param, T.minimum(0, param - learning_rate * gparam / (rho + (param_helper + gparam ** 2) ** 0.5) )  )) #adagrad
            updates.append((param, T.minimum(0,param - 1*learning_rate * gparam))) #no adagrad
        else:
            #updates.append((param, param - 1e2*learning_rate * gparam / (rho + (param_helper + gparam ** 2) ** 0.5)))
            updates.append((param, param - 1*learning_rate * gparam)) #no adagrad
        param_index += 1
    updates.append((iter_count, iter_count + 1))


    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index], outputs=cost,
            updates=updates,
            givens={
                x: data_set_x_train[index * batch_size:(index + 1) * batch_size],
                y: data_set_y_train[index * batch_size:(index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 100  # look as this many examples regardless
    #patience = train_set_x.get_value(borrow=True).shape[0] * n_epochs #no early stopping
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.99  # a relative improvement of this much is
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

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            print minibatch_avg_cost
 
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute absolute error loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
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

                    # test it on the test set
                    #test_losses = [test_model(i) for i
                    #               in [29]]
                    #test_score = numpy.mean(test_losses)
                    #for i in xrange(
                    #test_score, test_pred, test_actual,l1_in,l2_in = test_model(n_test_batches)

                    for i in xrange(n_valid_batches,n_valid_batches+n_test_batches):
                        test_losses_i, input_responses_i, test_pred_i, test_actual_i = test_model(i)
                        if i==n_valid_batches:
                            test_losses = test_losses_i
                            test_pred = test_pred_i
                            test_actual = test_actual_i
                            input_responses = input_responses_i
                        else:
                            test_losses += test_losses_i
                            test_pred += test_pred_i
                            test_actual += test_actual_i

                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i, test error of '
                           'best model %f') %
                          (epoch, minibatch_index + 1,
                           numpy.sum(test_score)))

            if patience <= iter:
                    done_looping = True
                    break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f'
           'obtained at iteration %i, with test performance %f') %
          (best_validation_loss, best_iter + 1, numpy.sum(test_score)))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
	
    #store data
    f = file('nat_parasols_sigUpdate' + '_lam_' + str(L1_reg) + '_nk_' + str(nkerns[0]) + '.save', 'wb')
    for obj in [params + [input_responses] + [test_score] + [test_losses] + [test_pred] + [test_actual]]:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

#loop over neurons and cross validation parameters
if __name__ == '__main__':
    SGD_training(L1_reg=0,L2_reg=0)


