# cifar
cifar training model using tensorflow

# Overview 
In this project, it build a model through tensorflow framework to learn cifar10 dataset.
There are 50,000 training images and 10,000 testing images in cifar10 with 32x32 size, 
  and is seperated into 10 classes.
This project simply implement a Convolutional Neural Network to learn these 10 categoreis.
The result can be seen using tensorboard by typing
'tensorboard --logdir /path/to/log'

# Requriement
- tensorflow 1.7 (cpu version)
- numpy
- matplotlib
- Kares

# code 
- cifar.py: CNN model to learn cifar10
- dataset.py: data manager of cifar data
- utils.py: one_hot function and image display function
- show_cifar_images.py: execution of displaying images
- hyper_cifar.py: multiparameters on cifar model
- hyper_cifar_main.py: execution of hyper_cifar.py
- stupid_hyper_main.py: stupid way to execute hyper_cifar.py

# Results
train accuracy: 90%
test accuracy: 68%
(overfitting)
