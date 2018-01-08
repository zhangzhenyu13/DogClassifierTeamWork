from Models.MLModel import *
import numpy as np
from DistanceDef.ClusterDistance import *
from LoadData.ReadData import *
import time
from sklearn.decomposition import PCA,FactorAnalysis

class ClusterModel_M(MLModel):
    def __init__(self, data):
        MLModel.__init__(self, data)
        self.clusters_means = {}
        self.clusters_V = {}
        self.reduction={}
        self.name='Maha'
        print(self.name,"method")

    def predict(self, X):
        'return distance'
        print("predict, size=",len(X))
        result = []
        count=0
        n=len(X)
        for x in X:

            d = np.zeros(shape=len(self.clusters_means.keys()),dtype=np.float32)
            #print("dog clusters")
            dogs=self.clusters_means.keys()

            for dog in dogs:
                #print(dog)
                x1=np.reshape(x, newshape=(1, x.size))
                #x1=self.reduction[dog].transform(x1)

                id=self.data.dog_class[dog]
                u = self.clusters_means[dog]
                covI=self.clusters_V[dog]
                #print(x.shape,cov.shape,u.shape)

                d1 = M_distance(x1, u,covI)
                d[id]=d1
            result.append(d)

            count=count+1
            print(count,"of",n)
        result=np.array(result)
        return result
    def train(self):
        print("building clustered classification model using Mahalanobis rule class")
        t1 = time.time()
        data = self.data
        dog_classes = data.uniqueLabels.keys()
        # get the data of the current class cluster
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

            #self.reduction[k]=FactorAnalysis(n_components=1000)
            #self.reduction[k].fit(X)
            #X=self.reduction[k].transform(X)
            X = np.mat(X)

            #print(X.shape,X.size)
            self.clusters_means[k] = Mean(X)
            self.clusters_V[k]=CovI(X)

        t2 = time.time()
        print("finished in", t2 - t1, "s")


class ClusterModel_Fisher(MLModel):
    def __init__(self,data):
        MLModel.__init__(self,data)
        self.clusters_means={}
        self.clusters_eigs=None
        self.name='Fisher'
        print(self.name,"method")
    def predict(self,X):
        print("predict, size=",len(X))
        A=self.clusters_eigs
        result=[]
        n=len(X)
        count=0
        for x in X:

            d=np.zeros(shape=len(self.clusters_means.keys()))
            dogs = self.clusters_means.keys()
            for dog in dogs:
                u=self.clusters_means[dog]
                u=np.reshape(np.array(u),u.size)
                d1=getDist(A, x, u)
                id=self.data.dog_class[dog]
                d[id]=d1
            result.append(d)
            count=count+1
            print(count, "of", n)
        return np.array(result)

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
        for k in self.clusters_means.keys():
            u=self.clusters_means[k]
            U.append(np.reshape(np.array(u),u.size))
        U=np.mat(U)
        B = B_fisher(U)
        A = A_eigenV(E.I * B)

        self.clusters_eigs=A
        t2=time.time()
        print("finished in",t2-t1,"s")


def vectorProbDst(A):
    for i in range(len(A)):
        a=A[i]
        min=np.min(a)
        a=a-min
        sum=np.sum(a)
        a=a/sum
        A[i]=a
    return A
#test for tuning
def test():
    data = FetchingData(image_folder='../data/outputJpg/', label_file='../data/originalData/labels.csv',split=0.9)
    learner = ClusterModel_M(data)
    learner.train()
    X, Y = data.getTestData()

    result = learner.predict(X)
    #result = data.StrDogArray(result)

    label_index = np.argmax(Y, 1)
    predict_index = np.argmin(result, 1)
    correct = np.sum(np.equal(label_index, predict_index))

    print("correct=", correct, "all=", data.testSize, "ratio=", correct / data.testSize)


if __name__ == '__main__':
    #test();exit(10)
    data = FetchingData(image_folder='../data/outputJpg/', label_file='../data/originalData/labels.csv')
    learner = ClusterModel_M(data)
    learner.train()
    testdata=TestData('../data/testOutput/')
    batchX,Id,count=testdata.getData(100)
    with open(file= "../data/result.csv",mode= "w",newline="") as f:
        writer=csv.writer(f)
        dogs=list(data.uniqueLabels.keys())
        dogs.sort()
        dogs.insert(0,'id')
        writer.writerow(dogs)
        while count>0:
            print("write a batch",count)
            #print(Id)
            Y=learner.predict(batchX)
            result=vectorProbDst(1/Y)

            result=np.array(result,dtype=np.str)
            result = np.insert(result, 0, Id, axis=1)
            #print(result)
            writer.writerows(result)
            f.flush()

            batchX,Id,count=testdata.getData(100)


