import tensorflow as tf
from hyper_cifar import CifarTrainModel

SUMMARY_DIR = 'D:/tmp/cifar' + '/hyper/6/'

CifarTrainModel (1e-3, True, True, SUMMARY_DIR+'lr_1e-3,fc=2,conv=2')

#CifarTrainModel (1e-3, True, False, SUMMARY_DIR+'lr_1e-3,fc=2,conv=1')

