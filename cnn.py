import tensorflow as tf
import numpy as np
from auxiliar_functions import *

def main():
	#Get the train data set
	X_train, Y_train, Y_train_i=Get_Train_Data()
	print("Training (X, Y, Y_ind) = ", X_train.shape, Y_train.shape, Y_train_i.shape)

	#Get the test data set (balanced)
	X_test, Y_test, Y_test_i=Get_Test_Data()
	print("Test (X, Y, Y_ind) = ", X_test.shape, Y_test.shape, Y_test_i.shape)

	#Total number of images
	tot_num_images=X_train.shape[0]+X_test.shape[0]
	print("Total Number of Images = ", tot_num_images)

	#N(number of observations), K(number of classes)
	N, K=Y_train_i.shape
	batch_size=50
	num_batches=int(N/batch_size)

	#First Convolutional Layer Initialization
	W1_shape=[5, 5, 1, 32]
	W1_init=Initial_Kernel(W1_shape)
	b1_init=Initial_Bias([W1_shape[3]])

	#Second Convolutional Layer Initialization
	W2_shape=[5, 5, 32, 64]
	W2_init=Initial_Kernel(W2_shape)
	b2_init=Initial_Bias([W2_shape[3]])

	#Densely Connected Layer Initialization
	W3_shape=[8*8*64, 1024]
	W3_init=Initial_Kernel(W3_shape)
	b3_init=np.zeros(W3_shape[1])

	#Output Layer Initialization
	W4_shape=[1024, K]
	W4_init=Initial_Kernel(W4_shape)
	b4_init=np.zeros(W4_shape[1])

	#Define the CNN variables
	X=tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 1), name='X')
	Y=tf.placeholder(dtype=tf.float32, shape=(None, K), name='Y')
	W1=tf.Variable(initial_value=W1_init, dtype=tf.float32, name='W1')
	b1=tf.Variable(initial_value=b1_init, dtype=tf.float32, name='b1')
	W2=tf.Variable(initial_value=W2_init, dtype=tf.float32, name='W2')
	b2=tf.Variable(initial_value=b2_init, dtype=tf.float32, name='b2')
	W3=tf.Variable(initial_value=W3_init, dtype=tf.float32, name='W3')
	b3=tf.Variable(initial_value=b3_init, dtype=tf.float32, name='b3')
	W4=tf.Variable(initial_value=W4_init, dtype=tf.float32, name='W4')
	b4=tf.Variable(initial_value=b4_init, dtype=tf.float32, name='b4')
	prob_keep=tf.placeholder(dtype=tf.float32)

	#First Convolutional and Max Pool Layer
	Z1=ConvLayer(X, W1, b1)
	Z1=MaxPool(Z1)

	#Second Convolutional and Max Pool Layer
	Z2=ConvLayer(Z1, W2, b2)
	Z2=MaxPool(Z2)
	#Dropout
	Z2_drop=tf.nn.dropout(Z2, prob_keep)
	Z2_shape=Z2_drop.get_shape()
	#Number of features=img_height*img_width*num_channels
	num_features=Z2_shape[1:4].num_elements()
	Z2_flat=tf.reshape(Z2_drop, [-1, num_features])

	#Densely Connected Layer (fully connected layer)
	Z3=tf.nn.relu(tf.matmul(Z2_flat, W3)+b3)

	#Output Layer (fully connected layer)
	Y_final=tf.matmul(Z3, W4)+b4
	Y_pred=tf.nn.softmax(Y_final)
	Y_pred_class=tf.argmax(Y_pred, axis=1)

	#Cost function (cross entropy)
	cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_final, labels=Y))

	#Forward propagation
	train=tf.train.AdamOptimizer(1e-4).minimize(cost)

	#Get best prediction
	correct_pred=tf.equal(Y_pred_class, tf.argmax(Y, axis=1))
	accuracy=tf.reduce_mean(tf.cast(correct_pred, 'float32'))

	prob_keep_drop=0.7

	#Create files to save the data
	stringpath="results_"+str(prob_keep_drop).split(sep='.')[0]+str(prob_keep_drop).split(sep='.')[1]
	train_acc_file=open(stringpath+"\\accuracy_train.txt", "w")
	val_acc_file=open(stringpath+"\\accuracy_validation.txt", "w")
	test_acc_file=open(stringpath+"\\accuracy_test.txt", "w")
	cost_file = open(stringpath+"\\cost.txt","w")

	#Perform the training
	session=tf.InteractiveSession()
	session.run(tf.global_variables_initializer())
	saver=tf.train.Saver()
	save_dir="C:\\Users\\User\\Desktop\\Code\\Tensorflow\\trained_cnn"
	#i (iterator for the number of epochs)
	for i in range(20):
		#n (iterator for the number of batches)
		for n in range(num_batches):
			print(n/num_batches*100, end="\r")
			#Get current batch
			X_batch=X_train[n*batch_size:(n*batch_size+batch_size), :, :, :]
			Y_batch_i=Y_train_i[n*batch_size:(n*batch_size+batch_size), :]
			#Forward propagation on current batch
			session.run(train, feed_dict={X:X_batch, Y:Y_batch_i, prob_keep:prob_keep_drop}) 

		cost_value=session.run(cost, feed_dict={X:X_batch, Y:Y_batch_i, prob_keep:1.0})
		accuracy_train=session.run(accuracy, feed_dict={X:X_train, Y:Y_train_i, prob_keep:1.0})
		accuracy_test=session.run(accuracy, feed_dict={X:X_test, Y:Y_test_i, prob_keep:1.0})
		prediction_test=session.run(Y_pred_class, feed_dict={X:X_test, Y:Y_test_i, prob_keep:1.0})
		#Save data to respective files
		train_acc_file.write(str(accuracy_train)+"\n")
		test_acc_file.write(str(accuracy_test)+"\n")
		cost_file.write(str(cost_value)+"\n")
		print("Iteration = ", i, "Cost = ", cost_value, "Train Accuracy = ", accuracy_train, "Test Accuracy = ", accuracy_test)
	saver.save(sess=session, save_path=save_dir)

if __name__=='__main__':
	main()