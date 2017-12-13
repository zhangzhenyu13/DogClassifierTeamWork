from LoadData.ReadData import *
import tensorflow as tf
import numpy as np
import time

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape,dtype=tf.float32)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x,W,strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
def max_pool_5x5(x):
    return tf.nn.max_pool(x, ksize=[1, 5, 5, 1],
                          strides=[1, 5, 5, 1], padding='SAME')
def normalize(x):
    return tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)



def convNetCls(data):
    learnRate = 1e-4
    batchSize = 50
    iterationNum = 1000
    data.batch=batchSize
    # picSize:xyz
    pX = 500
    pY = 500
    pC=3
    #para
#para
    W = {
        "k1": [5, 5, pC, 32],
        "k2": [5, 5, 32, 64],
        "k3": [5, 5, 64, 128],
        "k4": [5, 5, 128, 256],
        "w1": [5 * 5 * 256, 4096],
        "out": [4096, data.LabelCount]
    }
    B = {
        "b1": [32],
        "b2": [64],
        "b3": [128],
        "b4": [256],
        "b5": [4096],
        "out": [data.LabelCount]
    }

    #define dataGraph

    x = tf.placeholder(tf.float32, [None, pX * pY * pC])
    x_image = tf.reshape(x, [-1, pX, pY, pC])
    y = tf.placeholder(tf.float32, [None, data.LabelCount])


    W_conv1 = weight_variable(W["k1"])
    b_conv1 = bias_variable(B["b1"])
    W_conv2 = weight_variable(W["k2"])
    b_conv2 = bias_variable(B["b2"])
    W_conv3 = weight_variable(W["k3"])
    b_conv3 = bias_variable(B["b3"])
    W_conv4 = weight_variable(W["k4"])
    b_conv4 = bias_variable(B["b4"])

    W_fc1 = weight_variable(W["w1"])  # total connect from 2dArray to hidden layer perceptrons(1024)
    b_fc1 = bias_variable(B["b5"])
    W_fc2 = weight_variable(W["out"])  # connect from hiddenlayer to output layer(10)
    b_fc2 = bias_variable(B["out"])

    # Create the 1st conv layer, reduce to 100*100,32
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_5x5(h_conv1)
    # create the 2nd conv, layer reduce to 20*20,64
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_5x5(h_conv2)
    # Create the 3rd conv layer, reduce to 10*10,128
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    # create the 4th conv, layer reduce to 5*5,256
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4)
    # conv to multi-perceptrons layer, connect 5*5*256
    h_pool4_flat = tf.reshape(h_pool4, [-1, 5 * 5 * 256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
    # define dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # define output layer
    y_conv = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    # Define loss and optimizer

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
    train_step = tf.train.AdamOptimizer(learnRate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_conv, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)

    # Train
    print("runing convolutional networks")
    t1 = time.time()
    for i in range(1, iterationNum+1):
        batchX, batchY =data.getNextBatch()
        #print(batchX)
        #print(batchY)
        sess.run(train_step, feed_dict={x: batchX,
                                        y: batchY,
                                        keep_prob:0.5})
        if i % 20 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batchX, y: batchY,keep_prob:1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))

    t2 = time.time()
    print("finished in ",str(t2 - t1) + "s")



def run():
    data=FetchingData(image_folder='../data/tmpOutput/',label_file='../data/originalData/labels.csv',com=communitor)

    data.cache=0.2
    data.start()
    convNetCls(data)




if __name__=="__main__":
    run()
