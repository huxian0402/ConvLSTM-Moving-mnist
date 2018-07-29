#Moving MNIST dataset could be downloaded in the following link http://www.cs.toronto.edu/~nitish/unsupervised_video/
# convlstm(from RFL) + Moving Mnist
# hx
# 2018/7/3
# conv layer + tf. convlstm , only the first seq as train data
# input 10--1
# success, the best loss = 899.927

import tensorflow as tf
import numpy as np
from conv_lstm import BasicConvLSTMCell, InitLSTMSate
import config
from rnn import rnn, DropoutWrapper
import matplotlib.pyplot as plt
from PIL import Image
import visualization

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import scipy.io as sio

# convlstm
# input:(batch_size, time_steps, input_size_w, input_size_h, num_featmap)

def load_data(path):
    data = np.load(path)
    train = data[:, 0:7000, :, :]     # 7000 train
    test = data[:, 7000:10000, :, :]     # 7000 test   #<class 'tuple'>: (20, 3000, 64, 64)

    train_oneseq_data = []
    train_label = []
    train_oneseq = data[:, 0:1, :, :]  #(20,1,64,64)
    for i in range(10):    #(20,1,64,64) split  len = 10 seq
        aa = train_oneseq[i:i+10, :, :, :]
        train_oneseq_data.append(aa)
        train_label.append(train_oneseq[i+10:i+11, :, :, :])   #(1,1,64,64)
    #train = np.array(train_oneseq_data)   #shape  #(10,10,1,64,64)

    return train ,train_label,train_oneseq_data, test

height = 64
width = 64
time_steps = 10
is_train = True
#epoch = 1
batch_size = 1

# tensor
input_seq = tf.placeholder(tf.float32, [10, batch_size , height, width, 1])         # (10,batch_size ,64,64,1)
input_seq = tf.transpose(input_seq, perm=[1, 0, 2, 3, 4])
output_seq = tf.placeholder(tf.float32, [1, batch_size ,height, width, 1])         # (1, batch_size ,64,64,1)

# RNN_convLSTM  structure
def RNN_convLSTM(x ):

    #(1,19,64,64,1)     #num_filter, kernei_size
    x = tf.contrib.slim.conv2d(x, 1, 1, padding='SAME',activation_fn=tf.nn.relu)
    x = tf.contrib.slim.conv2d(x, 8, 1, padding='SAME', activation_fn=tf.nn.relu)
    x = tf.contrib.slim.conv2d(x, 16, 1, padding='SAME', activation_fn=tf.nn.relu)
    #(1,10,64,64,16)
                                      # [3 ,                       3]      ,                1024
    rnn_cell = BasicConvLSTMCell([64, 64], [config.conv_filter_size, config.conv_filter_size], config.hidden_size,
                                 is_train,
                                 forget_bias=1.0, activation=tf.nn.tanh)

    state_size = rnn_cell.state_size  #c(64,64,1024) ,h(64,64,1024)

    rnn_inputs = [tf.squeeze(input_, [1]) for input_ in
                  tf.split(axis=1, num_or_size_splits=time_steps, value=x)]   ##(8,1,64,64,16)

    init_state_net = InitLSTMSate({'input': rnn_inputs[0], 'state_size': state_size}, is_train)   # input': rnn_inputs[0] ???

    initial_state = init_state_net.get_output()

    rnn_inputs_new = rnn_inputs[1: time_steps]   #(10,1,64,64,1024)

    outputs, final_state, input_gates, forget_gates, output_gates \
        = rnn(rnn_cell, rnn_inputs_new, initial_state=initial_state)

    first_output = initial_state[1]
    outputs = [first_output] + outputs    #(10,1,64,64,1024)

    predictions_flat = tf.contrib.slim.conv2d(outputs[-1], 16, 3, padding='SAME', activation_fn=tf.nn.relu)
    predictions_flat = tf.contrib.slim.conv2d(predictions_flat, 8, 3, padding='SAME', activation_fn=tf.nn.relu)
    predictions_flat = tf.contrib.slim.conv2d(predictions_flat, 1, 1, padding='SAME', activation_fn=None)

    prediction = predictions_flat    #(1,1,64,64,1)

    return prediction

pred = RNN_convLSTM(input_seq )


# loss
loss_mse = tf.reduce_mean((output_seq - pred) ** 2)  # (1,1,64,64,1)
optimizer = tf.train.RMSPropOptimizer(0.005).minimize(loss_mse)   #lr = 0.01 very important!!!!

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    _batch_size = 1
    epoch = 50

    itra_batch = int( epoch * 10 / _batch_size)   # 7000

    # load all data
    # (20,7000,64,64) ,  10 (10,1,64,64)  ,(20, 3000, 64, 64)
    train_data, train_label,train_oneseq_data, test_data = load_data("mnist_test_seq.npy")

    # save all the frams of 2th sequence
    dir = './data-seq'
    j = 0
    loss_all = []
    itera_all =[]

    seq0 = train_data[:, j:j+1 , : , :]
    for i in range(20):
        ims = seq0[i:i+1,:,:,:] #(1,1,64,64)
        ims = np.squeeze(ims)   #(64 ,64)
        ims = Image.fromarray(ims)
        ims= ims.convert('RGB')
        save_dir = dir + '/data-seq{}_{}.png'.format(j,i )
        ims.save(save_dir)


    for i in range(itra_batch):

        # get next batch data
        train_data_batch = train_oneseq_data[i%10]   #<class 'tuple'>: (10, 1, 64, 64)

        train_data_input_batch = np.reshape(train_data_batch[0:10, :, :, :], [10, 1, 64, 64, 1])

        train_data_input_batch = np.transpose(train_data_input_batch, [1, 0, 2, 3, 4])  #(1, 10,  64, 64, 1)

        # get next batch label
        train_data_output_batch  = np.reshape(train_label[i%10], [1 ,1, 64, 64, 1])  #<class 'tuple'>: (10, 1, 64, 64, 1)

        sess.run(optimizer, feed_dict={input_seq: train_data_input_batch,        # batch input data
                                        output_seq: train_data_output_batch} )   # batch input label

        if (i + 1) % 1 == 0:
            preda, loss = sess.run([pred, loss_mse, ], feed_dict={input_seq: train_data_input_batch,
                                                                  output_seq: train_data_output_batch})

            #save loss ,hx
            save_fn = 'loss.mat'
            loss_all.append(loss)
            itera_all.append(i + 1)
            sio.savemat(save_fn,{'itera':itera_all , 'loss':loss_all})

            print('train_iteration', i + 1, 'loss:  ', loss)

            dir = './main-results'
            img_pred = np.squeeze(preda)  #(10,64,64)
            img_label = np.squeeze(train_data_output_batch)

            ims = img_label
            ims = np.squeeze(ims)  # (64 ,64)
            ims = Image.fromarray(ims)
            ims = ims.convert('RGB')
            save_dir = dir + '/label_{}.png'.format(i + 10)
            ims.save(save_dir)

            ims = img_pred
            ims = np.squeeze(ims)  # (64 ,64)
            ims = Image.fromarray(ims)
            ims = ims.convert('RGB')
            save_dir = dir + '/pred_{}.png'.format(i + 10)
            ims.save(save_dir)

    print("Optimization Finished!")