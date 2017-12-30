from LoadData.ReadData import *
from sklearn import svm
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn  import decomposition
from Models.MLModel import *
from sklearn import tree
import time
import numpy as np


class MyClassifier(MLModel):
    def __init__(self,data):
        MLModel.__init__(self,data)
        self.reduction=decomposition.FactorAnalysis(n_components=300)
        self.dogs=list(self.data.uniqueLabels.keys())
        self.dogs.sort()
        #print(self.dogs)
        self.model=[]
        for i in range(20):
            self.model.append(svm.LinearSVC())

    def predict(self,X):
        X=self.reduction.transform(X)
        y=[]
        for model in self.model:
            y.append(model.predict(X))

        #print(np.shape(y))
        y=np.array(y)
        y=np.transpose(y)
        #print(np.shape(y))

        y_d=[]

        for i in range(len(X)):
            y1=np.zeros(shape=len(self.dogs))
            for dog in y[i]:

                index=self.data.dog_class[dog]
                y1[index]=y1[index]+1
                #print("vote for",index,dog)
            #print(y1)
            y1=y1/np.sum(y1)
            y_d.append(y1)
            #print(y1)

        y_d=np.array(y_d)
        print(np.shape(y_d))

        return y_d

    def predictLabel(self,X):
        y_d=self.predict(X)
        y_max = np.argmax(y_d, axis=1)

        y = []
        for i in y_max:
            y.append(self.dogs[i])
        return y
    def train(self):

        X, Y = self.data.getXYlabels()
        print("training models,nm=", len(self.model))
        t1 = time.time()

        self.reduction.fit(X)
        X=self.reduction.transform(X)
        fold=len(self.model)
        fold_size=len(X)//fold
        for i in range(fold-1):
            print("fold:",i+1)
            model=self.model[i]

            m_X=np.concatenate((X[0:i*fold_size],X[(i+1)*fold_size:]),axis=0)
            m_Y=np.concatenate((Y[0:i*fold_size],Y[(i+1)*fold_size:]),axis=0)
            model.fit(m_X,m_Y)

        print("fold:",fold)
        m_X=X[0:(fold-1)*fold_size]
        m_Y=Y[0:(fold-1)*fold_size]
        self.model[fold-1].fit(m_X,m_Y)

        t2=time.time()

        print("finished training in",t2-t1,"s")
        X,Y=self.data.getXYtestlabels()
        if X is not None:
            Y_d=self.predict(X)
            Y1=self.predictLabel(X)
            #print("acc=",np.equal(Y,Y1))
            count=0
            for i in range(len(Y)):
                if Y[i]==Y1[i]:
                    count=count+1
                    print(Y1[i],"<--->",Y[i])
            print("correct sum=",count,"of",len(X))
            print((Y1!=Y).sum())

def test():
    data = FetchingData(image_folder='../data/outputJpg/', label_file='../data/originalData/labels.csv',split=0.98)

    learner = MyClassifier(data)
    learner.train()


if __name__ == '__main__':
    test()