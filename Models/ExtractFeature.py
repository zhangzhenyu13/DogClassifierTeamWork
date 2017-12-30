from LoadData.ReadData import *
from Models.MLModel import *
import tensorflow as tf
import xml.dom.minidom as xmlparser
import time
import numpy as np
import math
from Models.CNN import *

def runTuning():
    data = FetchingData(image_folder='../data/outputJpg/', label_file='../data/originalData/labels.csv',com=communitor,split=1.0)
    data.cache=0.01
    data.start()
    feadata=TestData('../data/testOutput/')
    feadata.addPics('../data/outputJpg/')
    learner=CNNFeatureExtractor(data,feadata)
    learner.train()
    data.batch=10
    tX,tY=data.getNextBatch()
    data.stop()

    X=learner.predict(tX)
    print("predict result=",len(X))
    print(X)
    print("original labels=",len(tX))
    print(tX)

class CNNFeatureExtractor(MLModel):

    def __init__(self,data,feadata):
        MLModel.__init__(self,data)
        self.fdata=feadata
        self.learningRate=1e-4
        self.iterNum=10000
        self.x = None
        self.y = None
        self.keep_prob=None
        self.sess=None
    # load net structure from file
    def readnetStructure(self,inDim, outDim):
        print("netWorks Structure")
        w = {}
        b = {}
        # READ PARAMETERS FROM ANNStructure.XML
        dom = xmlparser.parse("../data/models/NNStructure.xml")
        ANN = dom.documentElement
        prevNum = inDim
        nextNum = 0
        layer = ''
        i = 0
        while True:
            try:
                layer = 'L' + str(i)
                print(layer)
                nextNum = eval(ANN.getElementsByTagName(layer)[0].childNodes[0].nodeValue)
                w[i] = [prevNum, nextNum]
                b[i] = [nextNum]
                prevNum = nextNum
                i = i + 1
            except Exception as e:
                w[i] = [prevNum, outDim]
                b[i] = [outDim]
                break

        return (w, b)

    def predict(self, X):
        Y=self.sess.run(self.model,feed_dict={self.x: X, self.keep_prob: 1.0})
        Y=np.array(Y)
        return Y
    def CNNLayer(self,x_input,pX,pY,pC=3):
        print("shape=",pX,pY,pC)
        W = {
            "k1": [5, 5, 3, 32],
            "k2": [2, 2, 32, 64],
            "k3": [2, 2, 64, 128],
        }

        B = {
            "b1": [32],
            "b2": [64],
            "b3": [128],
        }

        # define dataGraph


        x_image = tf.reshape(x_input, [-1, pX, pY, pC])

        W_conv1 = weight_variable(W["k1"])
        b_conv1 = bias_variable(B["b1"])
        W_conv2 = weight_variable(W["k2"])
        b_conv2 = bias_variable(B["b2"])
        W_conv3 = weight_variable(W["k3"])
        b_conv3 = bias_variable(B["b3"])


        # Create the 1st conv layer, reduce to 20*20,32
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_5x5(h_conv1)
        # create the 2nd conv, layer reduce to 10*10,64
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        # Create the 3rd conv layer, reduce to 5*5,128
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)

        # conv to multi-perceptrons layer, connect 5*5*128

        h_pool_flat=tf.reshape(h_pool3, [-1, 5 * 5 * 128])

        inDim=5*5*128

        return h_pool_flat,inDim



    def train(self):
        data = self.data
        batchSize = data.batch
        data.batch = 1
        X, Y = data.getNextBatch()
        data.batch = batchSize
        # picSize:xyz
        dataLen = len(X[0])
        pC = 3

        pX = int(math.sqrt(dataLen/pC))
        pY = pX
        data=self.fdata
        #init para
        self.x = tf.placeholder(tf.float32, [None, dataLen])
        self.y = tf.placeholder(tf.float32, [None, dataLen])
        # drop out probability
        self.keep_prob = tf.placeholder(tf.float32)

        x,inDim=self.CNNLayer(self.x,pX,pY,pC)
        # define the Graph

        # def hidden layers
        w, b = self.readnetStructure(inDim, dataLen)  # defNetWork(inDim,outDim,hiddenLayer)
        W = {}
        B = {}
        H = {}
        for i in range(len(b)):
            # print(i)
            print(w[i])
            print(b[i])
            W[i] = tf.Variable(tf.truncated_normal(shape=w[i], stddev=0.1))
            B[i] = tf.Variable(tf.constant(value=0.1, shape=b[i]))

        H[0] = tf.nn.sigmoid(tf.matmul(x, W[0]) + B[0])
        for i in range(1, len(b) - 1):
            # print(i)
            H[i] = tf.nn.sigmoid(tf.matmul(H[i - 1], W[i]) + B[i])

        h_prev=H[len(b) - 2]
        model = tf.nn.dropout(h_prev, self.keep_prob)
        self.model = tf.matmul(model, W[len(b) - 1]) + B[len(b) - 1]

        loss = tf.reduce_mean(tf.square(self.model - self.y))
        train_step = tf.train.GradientDescentOptimizer(learning_rate=self.learningRate).minimize(loss)

        # begin train
        print("init variables")

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        print("running DCNN")
        t1 = time.time()

        batchSize=1000
        for i in range(1,self.iterNum+1):

            batchX, batchY,count = data.getData(batchSize)
            batchX=batchX[0:count]
            if count==0:
                data.pointer=0
                batchX, batchY, count = data.getData(batchSize)
                batchX = batchX[0:count]
            if count<batchSize:
                data.pointer=0

            self.sess.run(train_step, feed_dict={self.x: batchX, self.y: batchX, self.keep_prob: 0.5})
            if i % 100 == 0:
                print("step %d" % (i + 1))

                lossTrain = self.sess.run(loss, feed_dict={self.x: batchX, self.y: batchX, self.keep_prob: 1.0}) / len(batchX)
                print("train accuracy=%f" % (lossTrain))

        t2 = time.time()
        print("finished in", t2 - t1, "s")



if __name__ == '__main__':
    runTuning()
