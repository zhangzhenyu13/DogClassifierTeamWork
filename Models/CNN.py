from LoadData.ReadData import *
from Models.MLModel import *
import tensorflow as tf
import numpy as np
import time
import math
import sklearn.preprocessing as dataPre

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
                        strides=[1, 2, 2, 1], padding='VALID')
def max_pool_5x5(x):
    return tf.nn.max_pool(x, ksize=[1, 5, 5, 1],
                          strides=[1, 5, 5, 1], padding='SAME')
def normalize(x):
    return tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

class CNN(MLModel):
    def __init__(self,data):
        MLModel.__init__(self,data)
        self.learnRate = 1e-4
        self.iterationNum = 20000

        self.x=None
        self.y=None
        self.keep_prob=None
        self.sess=tf.InteractiveSession()

    def predict(self,x):
        result=self.sess.run(self.model,feed_dict={self.x:x,self.keep_prob:1.0})

        return np.array(result,dtype=str)
    def train(self):
        data = self.data
        batchSize = data.batch
        data.batch = 1
        X, Y = data.getNextBatch()
        data.batch = batchSize
        dataLen = len(X[0])
        pC = 3

        pX = int(math.sqrt(dataLen / pC))
        pY = pX

        #define dataGraph

        self.x = tf.placeholder(tf.float32, [None, pX * pY * pC])
        x_image = tf.reshape(self.x, [-1, pX, pY, pC])
        self.y = tf.placeholder(tf.float32, [None,len(data.uniqueLabels)])
        self.keep_prob = tf.placeholder(tf.float32)


        W_conv1 = weight_variable([2,2,pC,40])
        b_conv1 = bias_variable([40])
        W_conv2 = weight_variable([2,2,40,80])
        b_conv2 = bias_variable([80])
        W_conv3 = weight_variable([5,5,80,160])
        b_conv3 = bias_variable([160])
        W_conv4=weight_variable([2,2,160,200])
        b_conv4=bias_variable([200])



        W_fc1 = weight_variable([2*2*200,400])
        b_fc1 = bias_variable([400])
        W_fc2 = weight_variable([400,len(data.uniqueLabels)])
        b_fc2 = bias_variable([len(data.uniqueLabels)])

        # 1st cnn
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        # 2nd css
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        # 3rd cnn
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_5x5(h_conv3)
        # 4th cnn
        h_conv4=tf.nn.relu(conv2d(h_pool3,W_conv4)+b_conv4)
        h_pool4=max_pool_2x2(h_conv4)
        # conv to multi-perceptrons layer, connect 5*5*128
        h_pool4_flat = tf.reshape(h_pool4, [-1, 2 * 2 * 200])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
        # define dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        # define output layer
        self.model = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        max=tf.reduce_max(self.model,axis=1)
        min=tf.reduce_min(self.model,axis=1)
        max=tf.reshape(max,shape=(data.batch,1))
        min=tf.reshape(min,shape=(data.batch,1))
        ones=tf.ones(shape=(1,len(data.uniqueLabels)))

        max=tf.matmul(max,ones)
        min=tf.matmul(min,ones)

        self.model=(self.model-min)/(max-min)
        sum=tf.reduce_sum(self.model,axis=1)
        sum=tf.reshape(sum,shape=(data.batch,1))
        sum=tf.matmul(sum,ones)
        self.model=self.model/sum

        # Define loss

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.model))
        train_step = tf.train.AdagradOptimizer(self.learnRate).minimize(loss)


        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.model, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init = tf.global_variables_initializer()
        self.sess.run(init)

        tX,tY=data.getTestData()
        tX=tX[0:data.batch]
        tY=tY[0:data.batch]
        # Train
        print("runing convolutional networks")
        t1 = time.time()
        for i in range(1, self.iterationNum+1):
            batchX, batchY =data.getNextBatch()
            #print(batchX.shape)
            #print(batchY.shape)
            self.sess.run(train_step, feed_dict={self.x: batchX,
                                            self.y: batchY,
                                            self.keep_prob:0.5})

            if i % 20 == 0:
                print("step #%d"%(i))
                train_loss=self.sess.run(loss,feed_dict={
                    self.x:batchX,self.y:batchY,self.keep_prob:1.0
                })
                print("loss=",train_loss)
                train_accuracy = self.sess.run(accuracy,feed_dict={
                    self.x: batchX, self.y: batchY,self.keep_prob:1.0})
                print("training accuracy %g" % ( train_accuracy))


                test_accuracy = self.sess.run(accuracy, feed_dict={
                    self.x: tX, self.y: tY, self.keep_prob: 1.0})
                print("test accuracy %g" % (test_accuracy))
        t2 = time.time()
        print("finished in ",str(t2 - t1) + "s")



def runFortuning():
    data=FetchingData(image_folder='../data/outputJpg/',label_file='../data/originalData/labels.csv',com=communitor,split=0.9)

    data.cache=0.2
    data.start()
    learner=CNN(data)
    learner.train()
    data.stop()
    testdata=TestData('../data/testOutput/')
    with open("../data/result_cnn.csv","w",newline="") as f:
        writer=csv.writer(f)
        dogs = list(data.uniqueLabels.keys())
        dogs.sort()
        dogs.insert(0,'id')

        writer.writerow(dogs)

        tX, Id, count = testdata.getData(100)


        while count>0:
            result=learner.predict(tX)
            result=result[0:count]
            result=np.insert(result,0,Id,axis=1)
            writer.writerows(result)
            f.flush()
            if count<100:
                break
            tX,Id,count=testdata.getData(data.batch)



if __name__=="__main__":
    runFortuning()
