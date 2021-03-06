import threading
import random
import csv
import queue
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

communitor=threading.Condition()

#pipeline fetching data
class FetchingData(threading.Thread):
    def __init__(self,image_folder,label_file,com=None,split=1.0):

        # define a cache area for pipeline fetching data
        'cache 3 batch Size'
        threading.Thread.__init__(self)
        print("init data set")
        self.name=""

        self.labels={}#store the img id and label
        self.images=image_folder
        count = 0
        self.uniqueLabels={}
        with open(label_file, "r") as f:
            label_reader = csv.reader(f)
            header=True
            for row in label_reader:
                if header:
                    header=False
                    continue
                self.labels[row[0]]=row[1]

                if row[1] in self.uniqueLabels.keys():
                    self.uniqueLabels[row[1]].append(row[0])
                else:
                    self.uniqueLabels[row[1]]=[]
                    self.uniqueLabels[row[1]].append(row[0])
                count=count+1

                #if count>200:break  #for test purpose

        self.cache=0.2
        self.image_cache = queue.Queue()
        self.labels_cachae = queue.Queue()
        self.batch=100
        self.Size=count

        #define dog id in the order of its name
        self.dog_class={}
        dogs=list(self.uniqueLabels.keys())
        dogs.sort()

        for id in range(len(dogs)):
            self.dog_class[dogs[id]]=id
            #print(id,dogs[id])


        self.splitRatio = split

        self.testlabels={}
        count=0
        id=0
        for k in self.uniqueLabels.keys():

            id=id+1
            indexL=self.uniqueLabels[k]
            random.shuffle(indexL)
            num=int( (1-self.splitRatio) * len(self.uniqueLabels[k]) )
            for i in range(num):
                self.testlabels[indexL[i]]=self.labels[indexL[i]]
                self.labels.pop(indexL[i])
                count=count+1
        self.trainSize=self.Size-count
        self.testSize=count
        #communicator
        self.com = com
        self.putting_cache=False
        self.tag = True
        print("size for train and test =",self.trainSize,self.testSize,
              ",with",len(self.uniqueLabels.keys()),"different classes")

    # the stop, run, and getNextBatch method is in case we have too big data
    # we can get a pipeline handling that won't affect the train speed too much in this case
    def stop(self):
        self.com.acquire()
        self.tag=False
        self.com.notify()
        self.com.release()
    def run(self):
        cache=self.cache
        while self.tag:
            try:
                # once awake, require for lock to put data
                self.putting_cache=True
                print("putting more data to cache, ratio=",cache)
                while self.labels_cachae.qsize()<int(self.trainSize*cache):
                    indexL=self.labels.keys()
                    indexL=list(indexL)
                    random.shuffle(indexL)
                    #print(indexL)
                    size=int(self.Size*cache)-self.labels_cachae.qsize()
                    #print("need",size,"x")
                    for i in range(size):
                        dog=self.labels[indexL[i]]
                        #print(type(self.dog_class),dog)
                        dog_id=self.dog_class[dog]
                        x=np.zeros(len(self.uniqueLabels.keys()),dtype=np.float32)
                        x[dog_id]=1
                        self.labels_cachae.put(x)

                        with Image.open(self.images+indexL[i]+".jpg") as img:
                            pic = np.array(img.getdata(),dtype=np.float32)
                            pic = np.reshape(pic, newshape=pic.size)
                            self.image_cache.put(pic)
                            #print("type X=",type(pic))

                #after putting data, wait
                print("waiting... ")
                self.com.acquire()
                self.putting_cache=False
                self.com.wait()
                self.com.release()

            except Exception as e:
                print(e.args)
                print("error in run,end")
                self.tag=False
                self.putting_cache=False
        print("data object exits running")
    def getNextBatch(self):
        #print("in batch get")
        X=[]
        Y=[]

        if self.labels_cachae.qsize()<self.batch and self.putting_cache==False:
            self.com.acquire()
            print("less than 1 batch in cache")
            self.com.notify()
            self.com.release()

        for i in range(self.batch):
            X.append(self.image_cache.get())
            Y.append(self.labels_cachae.get())
            pass
        #print("read a batch")
        if self.labels_cachae.qsize()<3*self.batch and self.putting_cache==False:
            self.com.acquire()
            print("less than 2 batch in cache")
            self.com.notify()
            self.com.release()

        #print("init", np.shape(X), np.shape(Y))
        #print("Before type=", type(X), type(Y))
        X=np.array(X)
        Y=np.array(Y)
        X=np.reshape(X,newshape=(self.batch,len(X[0])))
        Y=np.reshape(Y,newshape=(self.batch,len(Y[0])))
        #print("After type=",type(X),type(Y))
        return (X,Y)

    def getXY(self):
        print("read all the train data")
        X=[]
        Y=[]
        for k in self.labels.keys():
            dog=self.labels[k]
            dog_id = self.dog_class[dog]
            x = np.zeros(len(self.uniqueLabels.keys()), dtype=np.float32)
            x[dog_id] = 1
            Y.append(x)
            with Image.open(self.images + k + ".jpg") as img:
                pic = np.array(img.getdata(), dtype=np.float32)
                pic = np.reshape(pic, newshape=pic.size)
                # print("type X=",type(pic))
                X.append(pic)
        X = np.array(X)
        Y = np.array(Y)
        X = np.reshape(X, newshape=(self.trainSize, len(X[0])))
        Y = np.reshape(Y, newshape=(self.trainSize, len(Y[0])))
        print("finished reading train all  the train data")
        return (X,Y)
    def getTestData(self):
        print("read all the test data")
        tX=[]
        tY=[]
        for k in self.testlabels.keys():
            dog=self.testlabels[k]
            dog_id = self.dog_class[dog]
            x = np.zeros(len(self.uniqueLabels.keys()), dtype=np.float32)
            x[dog_id] = 1
            tY.append(x)
            with Image.open(self.images + k + ".jpg") as img:
                pic = np.array(img.getdata(), dtype=np.float32)
                pic = np.reshape(pic, newshape=pic.size)
                #print("type X=",type(pic))
                tX.append(pic)
        tX = np.array(tX)
        tY = np.array(tY)
        print("finished reading test all  the test data")
        if len(tX)==0:
            return None,None
        return (tX,tY)
    def getXYlabels(self,batchSize=None):
        print("read all the train labeled data")

        if batchSize is None:
            batchSize=len(self.labels.keys())

        X=[]
        Y=[]
        for k in self.labels.keys():
            Y.append(self.labels[k])
            with Image.open(self.images + k + ".jpg") as img:
                pic = np.array(img.getdata(), dtype=np.float32)
                pic = np.reshape(pic, newshape=pic.size)
                # print("type X=",type(pic))
                X.append(pic)

        X=np.array(X)
        Y=np.array(Y)
        print("finished reading test all  the train labeled data,size=",len(Y))

        return X,Y


    def getXYtestlabels(self,batchSize=None):
        print("read all the test labeled data")

        if batchSize is None:
            batchSize = len(self.testlabels.keys())

        X = []
        Y = []
        for k in self.testlabels.keys():
            Y.append(self.testlabels[k])
            with Image.open(self.images + k + ".jpg") as img:
                pic = np.array(img.getdata(), dtype=np.float32)
                pic = np.reshape(pic, newshape=pic.size)
                # print("type X=",type(pic))
                X.append(pic)

        X = np.array(X)
        Y = np.array(Y)
        print("finished reading test all  the test labeled data,size=",len(Y))
        if len(X)==0:
            return (None,None)

        return X, Y

    def StrDogArray(self, dogs):
        Y=[]
        for dog in dogs:
            dog_id = self.dog_class[dog]
            x = np.zeros(len(self.uniqueLabels.keys()), dtype=np.float32)
            x[dog_id] = 1
            Y.append(x)
        Y=np.array(Y)
        #Y=np.reshape(Y,newshape=(len(dogs),len(Y[0])))
        return Y
#test data is very big, we need to fetch batch by batch
class TestData:
    def __init__(self,pic_folder):
        print("loading test dataset")
        import  os
        self.pointer=0
        files=os.listdir(pic_folder)
        self.images=[]
        count=0
        for file in files:
            if ".jpg" not in file:
                continue
            #key=file[:-4]
            self.images.append(pic_folder+file)
            count=count+1
            #if count>100:break #for quick test purpose
        self.Size=count
        random.shuffle(self.images)
        print("size=",self.Size)
    def addPics(self,pic_folder):
        print("adding test dataset")
        import os
        self.loadingFolder = pic_folder
        self.pointer = 0
        files = os.listdir(pic_folder)
        count = 0
        for file in files:
            if ".jpg" not in file:
                continue
            #key = file[:-4]
            self.images.append(pic_folder+file)
            count = count + 1
            # if count>100:break #for quick test purpose
        self.Size = self.Size+count
        random.shuffle(self.images)
        print("addings, size=", count,self.Size)
    def getData(self,batchSize=None):

        if batchSize is None:
            batchSize=self.Size
        X = []
        Id=[]
        print("read dataset with batch size=",batchSize)
        count=0
        while count<batchSize and self.pointer<self.Size:
            count=count+1

            imgf=self.images[self.pointer]
            left=imgf.rfind('/')
            right=imgf.rfind('.')
            id=imgf[left+1:right]
            Id.append(id)
            with Image.open(imgf) as img:
                pic = np.array(img.getdata())
                pic=np.reshape(pic,newshape=pic.size)
                X.append(pic)
            self.pointer = self.pointer + 1

        print("finished reading a batch of test dataset")
        return np.array(X),np.array(Id),count

#test
'learn how to extract data into a np.array'
def readJpg():
    img=Image.open('../data/outputjpg/000bec180eb18c7604dcecc8fe0dba07.jpg')

    plt.figure("dog")
    plt.imshow(img)
    plt.show()
    print(img)

    pic=np.array(img.getdata())
    print(pic[:10])
    print(pic.size,pic.shape,type(pic[0]),type(pic[0][0]))
    size=img.size
    color=len(img.getpixel((0,0)))
    pic=np.reshape(pic,newshape=pic.size)
    print(pic[0],pic.size,pic.shape,type(pic[0]))

'change the pictures into given size'
def transferJPG(images,width=500,height=500,srcDir="../data/originalData/train/",dstDir='../data/outputJpg/'):
    import os
    files = os.listdir(images)
    images=[]
    for file in files:
        if ".jpg" not in file:
            continue
        images.append(file)
    for image in images:
        with Image.open(srcDir+image) as img:
            x,y=img.size
            if x<y:
                padding=y-x
                box=[0,padding//2,x,y-padding//2]
                img=img.crop(box)
                pass
            elif x>y:
                padding=x-y
                box=[padding//2,0,x-padding//2,y]
                img=img.crop(box)
                pass
            else:
                pass
            newsize=(width,height)
            img=img.resize(newsize,Image.BILINEAR)
            img.save(dstDir+image)
def transferTestJpg(width=30,height=30):
    import os
    files=os.listdir('../data/originalData/test/')
    for file in files:
        if '.jpg' not in file:
            continue
        with Image.open('../data/originalData/test/'+file) as img:
            x,y=img.size
            if x<y:
                padding=y-x
                box=[0,padding//2,x,y-padding//2]
                img=img.crop(box)
                pass
            elif x>y:
                padding=x-y
                box=[padding//2,0,x-padding//2,y]
                img=img.crop(box)
                pass
            else:
                pass
            newsize=(width,height)
            img=img.resize(newsize,Image.BILINEAR)
            img.save('../data/testOutput/'+file)


if __name__ == '__main__':
    #readJpg()
    transferTestJpg(30,30)
    #exit(2)
    transferJPG("../data/originalData/train/",width=30,height=30)
    exit(1)
    '''
    data=FetchingData(image_folder='../data/outputJpg/',label_file='../data/originalData/labels.csv',com=communitor)

    data.cache=0.2
    data.batch=2
    data.start()

    for i in range(200):
        x,y=data.getNextBatch()
        print(np.shape(x), np.shape(y))
    data.getTestData()
    data.stop()'''
    testdata = TestData('../data/testOutput/')
    testdata.addPics('../data/outputJpg/')
    print(testdata.getData(50))
    print(testdata.getData(20))

