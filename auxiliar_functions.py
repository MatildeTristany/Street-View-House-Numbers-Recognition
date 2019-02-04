import numpy as np
from scipy.io import loadmat
from PIL import Image
import tensorflow as tf

#Processe the images (transform to greyscale)
def Preprocess(X):
    X=X.transpose((3, 0, 1, 2))
    X=np.asarray([np.asarray(Image.fromarray(X[i]), dtype=np.float32)/255 for i in range(len(X))])
    X=np.expand_dims(np.dot(X, [0.2989, 0.5870, 0.1140]), axis=3)
    return X

#Create a random subsample
def Subsample(X, Y, Y_i, n):
	Xsample=[]
	Ysample=[]
	Ysample_i=[]
	dim=X.shape
	ind=np.arange(dim[0])
	ind=np.random.choice(ind, size=n, replace=False)
	for i in ind:
		Xsample.append(X[i])
		Ysample.append(Y[i])
		Ysample_i.append(Y_i[i])
	return np.array(Xsample), np.array(Ysample), np.array(Ysample_i)

#Create a balanced subsample (each different classe has s representants in the subsample)
def Subsample_Balanced(X, Y, Y_i, n):
	Xsample=[]
	Ysample=[]
	Ysample_i=[]
	ind=[]
	for label in np.unique(Y):
		images=np.where(Y==label)[0]
		Yrandom_sample=np.random.choice(images, size=n, replace=False)
		ind=ind+Yrandom_sample.tolist()
	for i in ind:
		Xsample.append(X[i])
		Ysample.append(Y[i])
		Ysample_i.append(Y_i[i])
	return np.array(Xsample), np.array(Ysample), np.array(Ysample_i)	

#Do one hot encoding
def One_Hot(predicted_value):
    N=len(predicted_value)
    K=len(set(predicted_value))
    predicted_value_i=np.zeros([N, K])
    for i in range(N):
        if(predicted_value[i]==10):
            predicted_value_i[i, 0]=1
        else:
            predicted_value_i[i, predicted_value[i]]=1
    return predicted_value_i

#Gets the training data
def Get_Train_Data():
    Train_Data=loadmat('train_32x32.mat')
    X=Preprocess(Train_Data['X'])
    Y=np.asarray((Train_Data['y']).flatten(), dtype=np.int32)
    Y_i=One_Hot(Y)
    return Subsample(X, Y, Y_i, 30000)

#Gets the test data
def Get_Test_Data():
    Test_Data=loadmat('test_32x32.mat')
    X=Preprocess(Test_Data['X'])
    Y=np.asarray((Test_Data['y']).flatten(), dtype=np.int32)
    Y_i=One_Hot(Y)
    return Subsample_Balanced(X, Y, Y_i, 1595)

#Initialize kernel
def Initial_Kernel(shape):
    W=tf.truncated_normal(shape=shape, mean=0.0, stddev=0.1)
    return W

#Initialize bias
def Initial_Bias(shape):
    b=tf.constant(value=0.1, shape=shape)
    return b

#Do convolution
def ConvLayer(X, W, b):
    #Compute a 2D convolution layer given a 4D input and a kernel tensor
    #Padding SAME makes the output the same size as the input
    conv=tf.nn.conv2d(
        input=X, 
        filter=W, 
        strides=[1, 1, 1, 1], 
        padding='SAME')
    #Return rectified linear unit relu(x)=max(0,x)
    return tf.nn.relu(conv)

#Do Maxpooling
def MaxPool(conv):
    #Perform the maxpooling over 2*2 blocks
    pool=tf.nn.max_pool(
        value=conv,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME')
    return pool