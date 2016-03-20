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


### Get which experiment

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


basedir = '/vega/stats/users/erb2180/RetinaProject/Data/'
#basedir = '/Volumes/Backup/RetinaProject/Data/Data/'
dataset= ''+basedir+'NSEM'+exp+'_movResp.mat'
    
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
  
responses_test = responses_test2[:,cell_ind[0,:]==on_ind]
responses_train = responses_train2[:,cell_ind[0,:]==on_ind]

Ncells = responses_test.shape[1]
responses_test_shifted = numpy.transpose(responses_test)
batch_size = 3600
testResponses = numpy.zeros((Ncells,59,batch_size))
for i_trial in xrange(59):
    testResponses[:,i_trial,:] = responses_test_shifted[:,i_trial*batch_size:(i_trial+1)*batch_size]

fname = '/vega/stats/users/erb2180/RetinaProject_clean/TestResponses/NSEM_'+exp+'_'+cell_type+'_testResponses.npy'

numpy.save(fname,testResponses)


