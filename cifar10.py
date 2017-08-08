import tensorflow as tf
import numpy as np
import time
import os
from datetime import timedelta

IMAGE_DIM = 32#Num of rows / columns
IMAGE_DEPTH = 3#Number of channels
IMG_FLAT = IMAGE_DIM * IMAGE_DIM *IMAGE_DEPTH
IMG_SIZE = (32,32,3)#Image size as a tuple
global NUM_CLASSES 
NUM_CLASSES = 10
global BATCH_SIZE
BATCH_SIZE = 64

x_image = tf.placeholder(tf.float32,shape=[None,IMG_FLAT],name='images')
images = tf.reshape(x_image, shape=[-1, IMAGE_DIM,IMAGE_DIM,IMAGE_DEPTH])
y_ = tf.placeholder(tf.int32,shape=[None],name='labels')
y_actual = tf.one_hot(y_,depth=NUM_CLASSES)
def model(images):
    
    with tf.name_scope('conv_1') as scope:
        
        conv_weight1 = tf.get_variable(name = 'conv_weights1',shape=[3,3,3,16],initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                                       mode='FAN_IN',
                                                                                                       uniform=False,
                                                                                                       seed=None,
                                                                                                       dtype=tf.float32))
        conv = tf.nn.conv2d(images,conv_weight1,\
                             strides=[1,1,1,1],\
                             padding='SAME') 
        biases = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[16],name='bias1'))
        bias = tf.nn.bias_add(conv,biases)        
        conv1 = tf.nn.relu(bias,name=scope)
        
        
    pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],
                        strides=[1,2,2,1],
                        name='pool1',padding='SAME')

    with tf.name_scope('conv_2') as scope:
        
        conv_weight2 = tf.get_variable(name = 'conv_weights2',shape=[3,3,16,32],initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                                       mode='FAN_IN',
                                                                                                       uniform=False,
                                                                                                       seed=None,
                                                                                                       dtype=tf.float32))
        conv = tf.nn.conv2d(pool1,conv_weight2,\
                             strides=[1,1,1,1],\
                             padding='SAME') 
        biases = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[32],name='bias2'))
        bias = tf.nn.bias_add(conv,biases)        
        conv2 = tf.nn.relu(bias,name=scope)
        
        
    pool2 = tf.nn.max_pool(conv2,ksize=[1,3,3,1],
                        strides=[1,2,2,1],
                        name='pool2',padding='SAME') 
                        
    activations = tf.contrib.layers.flatten(pool2)

    with tf.name_scope('fc1') as scope:
                              
         fc_weight1 = tf.get_variable(name = 'fc_weights1',shape=[2048,64],initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                                       mode='FAN_IN',
                                                                                                       uniform=False,
                                                                                                       seed=None,
                                                                                                       dtype=tf.float32))
        
         fc1 = tf.matmul(activations,fc_weight1) 
         biases1 = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[64],name='fc1'))
         bias = tf.nn.bias_add(fc1,biases1)        
         fc_layer1 = tf.nn.relu(bias,name=scope)

    with tf.name_scope('fc2') as scope:
         fc_weight2 = tf.get_variable(name = 'fc_weights2',shape=[64,10],initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                                       mode='FAN_IN',
                                                                                                       uniform=False,
                                                                                                       seed=None,
                                                                                                       dtype=tf.float32))
        
         fc2 = tf.matmul(fc_layer1,fc_weight2) 
         biases2 = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[10],name='fc2'))
         bias = tf.nn.bias_add(fc2,biases2)        
         fc_layer2 = tf.nn.relu(bias,name=scope)
         
    return fc_layer2    
    
fc_layer2 = model(images)         
#cross_entropy loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_actual,
                                                            logits = fc_layer2))
train_net = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)                                                        
correct_prediction = tf.equal(tf.cast(tf.argmax(fc_layer2,1),dtype=tf.float32),y_actual)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

def optimize(iterations):    
         
         with tf.Session() as sess:
             sess.run(tf.initialize_all_variables())
             start_time = time.time()
             for i in range(iterations):
                 
                 images,labels = get_data(batch=i%4)
                 for j in range(images.shape[0] / BATCH_SIZE + 1):
                     
                    img = images[j*BATCH_SIZE:(j+1)*BATCH_SIZE,:]
                    L = labels[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
                    _,loss = sess.run([train_net,cross_entropy],feed_dict={x_image:img,y_:L})
                    print loss 
                    
             save_path = saver.save(sess, "/tmp/model.ckpt")
             print("Model saved in file: %s" % save_path)       
         end_time = time.time()
         
         # Difference between start and end-times.
         time_dif = end_time - start_time

         # Print the time-usage.
         print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

def get_data(batch,isTraining=True):

    ROOT_PATH = "/home/jay/Deep Network Structures/cifar-10-python"
    #for training and testing separately
    if isTraining:
        path = os.path.join(ROOT_PATH,"train_batches/")
    else:
        path = os.path.join(ROOT_PATH,"test_batches/")
    
    file_names = [f for f in os.listdir(path)]
    batch = unpickle(os.path.join(path,file_names[batch]))
    images,labels = np.array(batch['data'],np.float32),batch['labels']  
    #normalize the data
    images,labels = prepare_input(data=images,labels=labels)
    
    return images,np.array(labels,dtype=np.int32)

def prepare_input(data=None, labels=None):
    
    #do mean normaization across all samples
    mu = np.mean(data, axis=0)
    mu = mu.reshape(1,-1)
    sigma = np.std(data, axis=0)
    sigma = sigma.reshape(1, -1)
    data = data - mu
    data = data / sigma
    return data,labels
    
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
                                