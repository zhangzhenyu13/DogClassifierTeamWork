from Models.MLModel import *
import numpy as np
from DistanceDef.ClusterDistance import *
from LoadData.ReadData import *
import time


class ClusterModel_M(MLModel):
    def __init__(self, data):
        MLModel.__init__(self, data)
        self.clusters_means = {}
        self.clusters_V = {}

    def predict(self, X):

        result = []
        for x in X:
            d = np.inf
            dog_cls = ''
            for dog_class in self.clusters_means:
                u = self.clusters_means[dog_class]
                cov=self.clusters_V[dog_class]
                d1 = getDist(x, u,cov)
                if d1 < d:
                    d = d1
                    dog_cls = dog_class
            result.append(dog_cls)
        return result

    def train(self):
        print("building clustered classification model using Mahalanobis rule class")
        t1 = time.time()
        data = self.data
        dog_classes = data.uniqueLabels.keys()
        # get the data of the current class cluster
        E = None
        count = 0
        for k in dog_classes:
            print("progress...", count, "of", len(dog_classes), "for dog class=", k)
            count = count + 1
            dogs = data.uniqueLabels[k]
            X = []
            for dog in dogs:
                with Image.open(self.data.images + dog + ".jpg") as img:
                    pic = np.array(img.getdata(), dtype=np.float32)
                    pic = np.reshape(pic, newshape=pic.size)
                    X.append(pic)
            X = np.mat(X)
            # print(X.shape,X.size)
            self.clusters_means[k] = Mean(X)
            self.clusters_V[k]=Cov(X)

        t2 = time.time()
        print("finished in", t2 - t1, "s")


class ClusterModel_Fisher(MLModel):
    def __init__(self,data):
        MLModel.__init__(self,data)
        self.clusters_means={}
        self.clusters_eigs=None
    def predict(self,X):

        A=self.clusters_eigs
        result=[]
        for x in X:
            d=np.inf
            dog_cls=''
            for dog_class in self.clusters_means:
                u=self.clusters_means[dog_class]
                d1=getDist(A, x, u)
                if d1<d:
                    d=d1
                    dog_cls=dog_class
            result.append(dog_cls)
        return result

    def train(self):
        print("building clustered classification model using fisher's rule class")
        t1=time.time()
        data=self.data
        dog_classes=data.uniqueLabels.keys()
        #get the data of the current class cluster
        E=None
        count=0
        for k in dog_classes:
            print("progress...",count,"of",len(dog_classes),"for dog class=",k)
            count=count+1
            dogs=data.uniqueLabels[k]
            X=[]
            for dog in dogs:
                with Image.open(self.data.images + dog + ".jpg") as img:
                    pic = np.array(img.getdata(), dtype=np.float32)
                    pic = np.reshape(pic, newshape=pic.size)
                    X.append(pic)
            X=np.mat(X)
            #print(X.shape,X.size)
            self.clusters_means[k]=Mean(X)
            if E is None:
                E=Cov(X)
            else:
                E=E+Cov(X)
        print("Final Step")
        U=[]
        for u in self.clusters_means:
            U.append(np.reshape(np.array(u),u.size))
        U=np.mat(U)
        B = B_fisher(U)
        A = A_eigenV(E.I * B)

        self.clusters_eigs=A
        t2=time.time()
        print("finished in",t2-t1,"s")

if __name__ == '__main__':
    data=FetchingData(image_folder='../data/outputJpg/',label_file='../data/originalData/labels.csv')
    learner=ClusterModel_M(data)
    learner.train()
    X,Y=data.getTestData()
    X=np.mat(X)
    result=learner.predict(X)
    result=data.StrDogArray(Y)
    same=np.equal(Y,result)
    print(same,np.sum(same))
