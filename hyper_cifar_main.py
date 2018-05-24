import tensorflow as tf
from hyper_cifar import CifarTrainModel

SUMMARY_DIR = 'D:/tmp/cifar' + '/hyper/7/'

def make_hparam_string (learning_rate, use_two_fc, use_two_conv):
    fc = 2 if use_two_fc else 1
    conv = 2 if use_two_conv else 1
    return "lr_{},fc={},conv={}".format(learning_rate, fc, conv)

# Try a few learning rates
for learning_rate in [1e-3, 1e-4, 1e-5]:

    # Try a model with fewer layers
    for use_two_fc in [True, False]:
        for use_two_conv in [True, False]:

            # Construct a hyperparameter string for each one (exmaple: lr_1e-3,fc=2,conv=2)
            hparam_str = make_hparam_string (learning_rate, use_two_fc, use_two_conv)
            summary_dir = SUMMARY_DIR + hparam_str

            print ("hparam_str :{}".format(summary_dir))

            # Actually run with the new settings
            CifarTrainModel (learning_rate, use_two_fc, use_two_conv, summary_dir)

