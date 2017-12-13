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
    def __init__(self,image_folder,label_file,com):

        # define a cache area for pipeline fetching data
        'cache 3 batch Size'
        threading.Thread.__init__(self)
        print("init data set")
        self.name=""

        self.labels={}
        self.images=image_folder
        count = 0
        uniqueLabels=set()
        with open(label_file, "r") as f:
            label_reader = csv.reader(f)
            header=True
            for row in label_reader:
                if header:
                    header=False
                    continue
                self.labels[row[0]]=row[1]
                uniqueLabels.add(row[1])
                count=count+1




        self.cache=0.5
        self.image_cache = queue.Queue()
        self.labels_cachae = queue.Queue()
        self.batch=100
        self.Size=count
        self.LabelCount=len(uniqueLabels)
        self.dog_class={}
        id=0
        for dog in uniqueLabels:
            self.dog_class[dog]=id
            id=id+1

        #communicator
        self.com = com
        self.putting_cache=False
        self.tag = True
        print("size=",self.Size,",with",self.LabelCount,"different classes")

    # the stop, run, and getNextBatch method is in case we have too big data
    # we can get a ppeline handling that won't affect the train speed too much in this case
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
                self.com.acquire()
                self.putting_cache=True
                print("putting more data to cache, ratio=",cache)
                while self.labels_cachae.qsize()<int(self.Size*cache):
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
                        x=np.zeros(self.LabelCount,dtype=np.float32)
                        x[dog_id]=1
                        self.labels_cachae.put(x)

                        with Image.open(self.images+indexL[i]+".jpg") as img:
                            pic = np.array(img.getdata(),dtype=np.float32)
                            pic = np.reshape(pic, newshape=pic.size)
                            self.image_cache.put(pic)
                            #print("type X=",type(pic))

                #after putting data, wait
                print("waiting... ")
                self.putting_cache=False
                self.com.wait()
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
        print("read a batch")
        if self.labels_cachae.qsize()<3*self.batch and self.putting_cache==False:
            self.com.acquire()
            print("less than 2 batch in cache")
            self.com.notify()
            self.com.release()

        #print("init", np.shape(X), np.shape(Y))
        #print("Before type=", type(X), type(Y))
        X=np.reshape(X,newshape=(self.batch,len(X[0])))
        Y=np.reshape(Y,newshape=(self.batch,len(Y[0])))
        #print("After type=",type(X),type(Y))
        return (X,Y)

#test data is very big, we need to fetch batch by batch
class TestData:
    def __init__(self,pic_folder):
        print("loading test data")
        import  os
        self.loadingFolder=pic_folder
        self.pointer=0
        files=os.listdir(pic_folder)
        self.images=[]
        count=0
        for file in files:
            if ".jpg" not in file:
                continue
            key=file[:-4]
            self.images.append(key)
            count=count+1
        self.Size=count
        print("size=",self.Size)
    def getData(self,batchSize=None):

        count=0
        if batchSize is None:
            batchSize=self.Size
        X = np.zeros(shape=batchSize)

        for key in self.images:
            with Image.open(self.loadingFolder+key+".jpg") as img:
                pic = np.array(img.getdata())
                pic=np.reshape(pic,newshape=pic.size)
                X[count]=pic
                count=count+1
                self.pointer=self.pointer+1
            if count>=batchSize or self.pointer>=self.Size:
                break
        return X

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
def transferJPG(images,width=500,hight=500):
    import os
    files = os.listdir(images)
    images=[]
    for file in files:
        if ".jpg" not in file:
            continue
        images.append(file)
    for image in images:
        with Image.open("../data/originalData/train/"+image) as img:
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
            newsize=(width,hight)
            img=img.resize(newsize,Image.BILINEAR)
            img.save('../data/tmpOutput/'+image)

if __name__ == '__main__':
    #readJpg()
    #transferJPG("../data/originalData/train/")
    #exit(1)
    testdata=TestData('../data/originalData/test/')
    data=FetchingData(image_folder='../data/outputjpg/',label_file='../data/originalData/labels.csv',com=communitor)


    data.cache=0.01
    data.batch=2
    data.start()
    for i in range(100):
        x,y=data.getNextBatch()
        print(np.shape(x), np.shape(y))

    data.stop()

