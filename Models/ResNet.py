from torchvision import models,transforms
from torch import nn,optim
from torch.nn import functional
import torch
from torch.autograd import Variable
from Models.MLModel import *
from torch.utils import data
import csv
from PIL import Image
import numpy as np
import math
import sklearn.preprocessing as dataPre
class MyDataSet(data.Dataset):
    def __init__(self,img_folder,label_file,img_transform,target_transform):
        self.label_list=[]
        self.img_list=[]
        self.target_transorfm=target_transform
        self.img_transform=img_transform
        self.dogs=set()
        import os
        files = os.listdir(img_folder)
        for file in files:
            if '.jpg' not in file:
                continue
            self.img_list.append(file[:-4])


        if label_file is not None:
            labels={}
            with open(label_file, "r") as f:
                label_reader = csv.reader(f)
                header = True
                for row in label_reader:
                    if header:
                        header = False
                        continue
                    labels[row[0]] = row[1]
                    self.dogs.add(row[1])
            #add label str
            for file in self.img_list:
                self.label_list.append(labels[file])
            #change label str to arr
            self.dogs = list(self.dogs)
            self.dogs.sort()
            #print(self.dogs)
            for i in range(len(self.label_list)):
                dog = self.label_list[i]
                index = self.dogs.index(dog)
                self.label_list[i] = index
                #print("index of",dog,"is",index)

        for i in range(len(self.img_list)):
            self.img_list[i]=img_folder+self.img_list[i]+".jpg"

    def __getitem__(self, index):
        img=self.img_list[index]

        with Image.open(img) as pic:
            img=np.array(pic.getdata())

        w=int(math.sqrt(img.size//3))
        img=np.reshape(img,newshape=(w,w,3))
        #print(np.shape(img))

        if self.img_transform is not None:

            img=self.img_transform(img)
            pass

        #for test
        if len(self.label_list)==0:
            r=self.img_list[index].rfind('/')
            label=self.img_list[index][r+1:-4]
            return img,label

        #for train
        label=self.label_list[index]

        if self.target_transorfm is not None:
            label=self.target_transorfm(label)
            pass

        return img,label

    def __len__(self):
        return  len(self.img_list)

kwargs = {'num_workers': 1, 'pin_memory': True}

def vectorProbDst(A):
    for i in range(len(A)):
        a=A[i]
        min=np.min(a)
        a=a-min
        sum=np.sum(a)
        a=a/sum
        A[i]=a
    return A

class MyResNet(MLModel):
    def loadData(self):
        self.train_loader = data.DataLoader(
            self.traindata,
            batch_size=50, shuffle=True, **kwargs)

        self.test_loader = data.DataLoader(
            self.testdata,
            batch_size=100, shuffle=True, **kwargs)


    def __init__(self,data):
        MLModel.__init__(self,data)
        self.traindata=data[0]
        self.testdata=data[1]
        self.loadData()

        self.model=models.resnet152(pretrained=True)
        self.learningRate=1e-4
        self.iterNum=1000

        self.model.fc = nn.Linear(in_features=2048, out_features=120)
        self.loss = nn.CrossEntropyLoss()

        #self.model.cuda()

    def predict(self,X):

        self.model.eval()
        print("predict and write result to file")

        with open("../data/result_resNet.csv", "w", newline="") as f:
            writer = csv.writer(f)
            dogs=self.traindata.dogs
            dogs.insert(0, 'id')

            writer.writerow(dogs)

            for step,(data,id) in enumerate(self.test_loader):
                #data = data.cuda()
                data = Variable(data, volatile=True)
                output = self.model(data)
                # pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                result = output.data.numpy()
                result=vectorProbDst(result)
                result=np.array(result,dtype=np.str)
                result = np.insert(result, 0, id, axis=1)
                writer.writerows(result)
                f.flush()
                if step % 20 == 0:
                    print('Test #{}: [{}/{} ({:.0f}%)]'.format(step,
                         step * len(data), len(self.test_loader.dataset),100. * step / len(self.test_loader)
                    ) )


    def train(self):

        train_step=optim.SGD(params=[self.model.fc.weight,self.model.fc.bias],lr=self.learningRate)
        self.model.train()
        print("training model")
        self.model.train()
        for epoch in range(self.iterNum):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data,target=data,torch.from_numpy(np.array(target,dtype=np.long))
                #data, target = data.cuda(), target.cuda()

                data, target = Variable(data), Variable(target)
                train_step.zero_grad()
                output = self.model(data)
                #loss = functional.nll_loss(output, target)
                loss=self.loss(output,target)
                loss.backward()
                train_step.step()
                if batch_idx % 20 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                               100. * batch_idx / len(self.train_loader), loss.data[0]))
                if loss.data[0]<1:break

if __name__=="__main__":
    #print(models.resnet18())
    traindata=MyDataSet(img_folder='../data/outputJpg/', label_file='../data/originalData/labels.csv',
              img_transform=transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
              ]), target_transform=None)
    testdata=MyDataSet(img_folder='../data/testOutput/', label_file=None,
              img_transform=transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
              ]), target_transform=None)


    model=MyResNet((traindata,testdata))
    #model.train()
    result=model.predict(None)
