import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from auxiliar_functions import *

test=loadmat('test_32x32.mat')
train=loadmat('train_32x32.mat')

testLabels=np.array(test['y'].flatten())
trainLabels=np.array(train['y'].flatten())

for i in range(len(testLabels)):
	if(testLabels[i]==10):
		testLabels[i]=0
	if(trainLabels[i]==10):
		trainLabels[i]=0

testLabels_bal=Get_Train_Data()[1]
trainLabels_bal=Get_Test_Data()[1]

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
fig.suptitle('Class Distribution Unbalanced', fontsize=14, fontweight='bold', y=1.05)
ax1.hist(trainLabels, bins=10)
ax1.set_title("Training set")
ax1.set_xlim(0, 9)
ax2.hist(testLabels, color='g', bins=10)
ax2.set_title("Test set")
fig.tight_layout()

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
fig.suptitle('Class Distribution Balanced', fontsize=14, fontweight='bold', y=1.05)
ax1.hist(trainLabels_bal, bins=10)
ax1.set_title("Training set")
ax1.set_xlim(0, 9)
ax2.hist(testLabels_bal, color='g', bins=10)
ax2.set_title("Test set")
fig.tight_layout()