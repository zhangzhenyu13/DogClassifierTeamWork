import numpy as np
import math
def Covariance(x1,x2):
    x1=np.array(x1)
    x2=np.array(x2)

    x1=np.reshape(x1,newshape=x1.size)
    x2=np.reshape(x2,newshape=x2.size)

    mean1 = np.mean(x1)
    mean2 = np.mean(x2)
    cov = np.dot(x1-mean1,x2-mean2)

    return cov
#this function is aborted due to its low efficiency
def CovData(X):

    row,col=X.shape
    X=X.getT()
    V=np.zeros(shape=[col,col],dtype=np.float32)
    for i in range(col):
        for j in range(i,col):
            V[i,j]=Covariance(X[i],X[j])
            V[j,i]=V[i,j]

    return np.mat(V)/(X.size-col)

def CovI(X):
    # numpy cov function assume that each colume is a sample if rowvar not set to False
    #print("cov",X.shape)
    cov = np.mat(np.cov(X, rowvar=False))
    return cov.I
def Mean(X):
    mean = np.mean(X, axis=0)
    return mean
def M_distance(x,mean,covI):
    '''
    :param x: an instance from G or else
    :param mean: the mean of data cluster
    :param cov: the covriance of cluster
    :return: the Mahalanobis distance of x to G's center
    '''
    x=np.mat(x)
    m=(x-mean)*covI*(x-mean).getT()
    m=np.reshape(np.array(m),m.size)

    d= math.fabs(m[0])

    return d

'''
due to we lack the information about the distribution of the density or P function of 
the data we are about to handle later, we don't use this Bayes Mathod to find the G of x
'''
# bayes method for divide sample
def LossC(m='classNum'):
    #default method
    C=np.zeros(shape=(m,m),dtype=np.float32)
    for i in range(m):
        for j in range(i,m):
            C[i,j]=1.0
            C[j,i]=C[i,j]
    return C
def ECM_Func(Q,P):
    '''

    :param Q: for each i, the pre P that it belong to G_i, a list
    :param P: for eahc i, judgement Probability  that it belong to j, a matrix
    :return: total loss in current divide, ecm
    '''
    ecm=0.0
    m=Q.size
    for i in range(m):
        q=Q[i]
        loss=0.0
        for j in range(m):
            loss=loss+P[i,j]
        ecm=ecm+loss*q

    return ecm

#fisher's method
def get_m_center(X_list):
    U = []

    for i in range(len(X_list)):
        #print(X_list[i])
        #print (Mean(X_list[i]))
        u=Mean(X_list[i])
        #print('u=',u)
        u=np.reshape(np.array(u),u.size)
        #print(u)
        U.append(u)
    U = np.mat(U)

    return U
def B_fisher(U):
    '''
    :param U: array of len=m with matrix of corresponding X
    :return: matrix B
    '''

    avg=np.mean(U)
    B = (U[0] - avg).T * (U[0] - avg)
    for i in range(1,len(U)):
        B=B+(U[i]-avg).T * (U[i]-avg)
    return B

def E_fisher(X_list):
    '''
    :param X_list: list of len=m with corresponding matrix of X
    :return: matrix E of sum V
    '''
    E=Cov(X_list[0])
    for i in range(1,len(X_list)):
        E=E+Cov(X_list[i])
    return E

def A_eigenV(mat1):
    f,a=np.linalg.eig(mat1)
    A={}

    for i in range(f.size):
        a1=np.array(a[i])
        A[f[i]]=np.reshape(a1,a1.size)
    #print("A=",A)
    f=np.sort(f,axis=None)
    #print(f)
    #print("len f",len(f))
    eig=[]
    for i in range(f.size):
        if f[f.size-1-i]<0.01*f[f.size-1]: break
        eig.append(A[f[f.size-1-i]])
    #print("eig=",eig)
    print("len a",len(eig))
    return np.array(eig)
def getDist(A,x,u):
    '''
    :param a: eigen vectors, np.array
    :param x: a sample ,np.array
    :param u: center of a given class, np.array
    :return: distance between a and u_class, float
    '''
    #print(A.shape,x.shape,u.shape)
    d=0
    u=np.reshape(np.array(u),u.size)
    for i in range(len(A)):
        #print(a[i].shape,x.shape,u.shape)
        a=A[i]
        a=np.reshape(np.array(a),a.size)
        d=d+math.fabs(np.dot(a,x)-np.dot(a,u))
    return d
#test
if __name__ == '__main__':
    print()
    G=[[1,0,0],[4,5,6],[0,0,9],[10,11,12]]
    x=[1,0,0]
    G=np.mat(G)
    G2=G+10
    x=np.array(x)
    mean,cov=(Mean(G),Cov(G))
    print(M_distance(x,mean,cov))
    #fisher
    X_list=[]
    X_list.append(G)
    X_list.append(G2)
    #print(X_list)
    U=get_m_center(X_list)
    #print('U=',U)
    B=B_fisher(U)
    #print('B=',B)
    E=E_fisher(X_list)
    #print('E=',E)
    A=A_eigenV(E.I*B)
    #print('A=',A)
    print(type(A),A)
    print(type(U),U)
    print(type(np.array(U)),np.array(U))
    for u in U:
        print(getDist(A,x,u))
