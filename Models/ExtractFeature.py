from LoadData.ReadData import *
from Models.MLModel import *
import tensorflow as tf
import xml.dom.minidom as xmlparser
import time
import numpy as np
def runTuning():
    data = FetchingData(image_folder='../data/outputJpg/', label_file='../data/originalData/labels.csv',com=communitor)
    data.cache=0.2
    data.start()
    learner=NN(data)
    learner.train()
    X=learner.predict(data.getTestData())
    data.stop()
class NN(MLModel):

    def __init__(self,data):
        MLModel.__init__(self,data)
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
        dom = xmlparser.parse("..\data\models\\NNStructure.xml")
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
    def train(self):
        data=self.data
        X, Y = data.getNextBatch()
        # def Learning parameters
        inDim = len(X[0])
        outDim = len(Y[0])
        # define the Graph
        self.x = tf.placeholder(tf.float32, [None, inDim])
        self.y = tf.placeholder(tf.float32, [None, inDim])
        # drop out probability
        self.keep_prob = tf.placeholder(tf.float32)
        # def hidden layers
        w, b = self.readnetStructure(inDim, inDim)  # defNetWork(inDim,outDim,hiddenLayer)
        W = {}
        B = {}
        H = {}
        for i in range(len(b)):
            # print(i)
            print(w[i])
            print(b[i])
            W[i] = tf.Variable(tf.truncated_normal(shape=w[i], stddev=0.1))
            B[i] = tf.Variable(tf.constant(value=0.1, shape=b[i]))

        H[0] = tf.nn.sigmoid(tf.matmul(self.x, W[0]) + B[0])
        for i in range(1, len(b) - 1):
            # print(i)
            H[i] = tf.nn.sigmoid(tf.matmul(H[i - 1], W[i]) + B[i])

        h_prev=H[len(b) - 2]
        model = tf.nn.dropout(h_prev, self.keep_prob)
        self.model = tf.matmul(model, W[len(b) - 1]) + B[len(b) - 1]

        loss = tf.reduce_mean(tf.square(self.model - self.y))
        train_step = tf.train.GradientDescentOptimizer(learning_rate=self.learningRate).minimize(loss)
        # test data
        tX, tY = data.getTestData()
        # begin train
        print("init variables")

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        print("running multi-layer perceptrons")
        t1 = time.time()

        for i in range(self.iterNum):
            batchX, batchY = data.getNextBatch()
            self.sess.run(train_step, feed_dict={self.x: batchX, self.y: batchX, self.keep_prob: 0.5})
            if i % 20 == 0:
                print("step %d" % (i + 1))
                lossTest = self.sess.run(loss, feed_dict={self.x: tX, self.y: tX, self.keep_prob: 1.0}) / data.testSzie
                print("test accuracy=%f" % (lossTest))
                lossTrain = self.sess.run(loss, feed_dict={self.x: batchX, self.y: batchX, self.keep_prob: 1.0}) / len(batchX)
                print("train accuracy=%f" % (lossTrain))

        t2 = time.time()
        print("finished in", t2 - t1, "s")



if __name__ == '__main__':
    runTuning()