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
#weight decay
WEIGHT_DECAY = 0.005
NUM_BATCHES=5 
#model stored at
save_path = '/home/jay/Deep_Structures/TF/my_test_model'
#log_dir
log_dir = '/home/jay/Deep_Structures/Summary/'

def model(_Xs_images,_Ys_labels,keep_prob,weights={}):
    
    with tf.name_scope('Reshape'):    
        _Xs = tf.reshape(_Xs_images, shape=[-1, IMAGE_DIM,IMAGE_DIM,IMAGE_DEPTH]) 
        _Ys = tf.one_hot(_Ys_labels,depth=NUM_CLASSES)
    
    with tf.name_scope('conv_1'):
        
        conv_weight1 = tf.get_variable(name = 'conv_weights1',shape=[3,3,3,16],initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                                       mode='FAN_IN',
                                                                                                       uniform=False,
                                                                                                       seed=None,
                                                                                                       dtype=tf.float32))
        weights['conv_weights1'] = conv_weight1                                                                                               
        conv = tf.nn.conv2d(_Xs,conv_weight1,\
                                strides=[1,1,1,1],\
                                padding='SAME') 
        biases = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[16],name='bias_1'))
        bias = tf.nn.bias_add(conv,biases)        
        conv1 = tf.nn.relu(bias,name='ReLU_Conv_1')
        pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],\
                               strides=[1,2,2,1],\
                               name='pool_1',padding='SAME')
        weight_summaries(weights['conv_weights1'])
        
    with tf.name_scope('conv_2'):
        
        conv_weight2 = tf.get_variable(name = 'conv_weights2',shape=[3,3,16,32],initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                                       mode='FAN_IN',
                                                                                                       uniform=False,
                                                                                                       seed=None,
                                                                                                       dtype=tf.float32))
        weights['conv_weights2'] = conv_weight2                                                                                                
        conv = tf.nn.conv2d(pool1,conv_weight2,\
                             strides=[1,1,1,1],\
                             padding='SAME') 
        biases = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[32],name='bias_2'))
        bias = tf.nn.bias_add(conv,biases)        
        conv2 = tf.nn.relu(bias,name='ReLU_Conv_2')
        pool2 = tf.nn.max_pool(conv2,ksize=[1,3,3,1],\
                             strides=[1,2,2,1],\
                             name='pool_2',padding='SAME') 
        weight_summaries(weights['conv_weights2'])
                
    activations = tf.contrib.layers.flatten(pool2)

    with tf.name_scope('fc_1'):
                              
         fc_weight1 = tf.get_variable(name = 'fc_weights1',shape=[2048,64],initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                                       mode='FAN_IN',
                                                                                                       uniform=False,
                                                                                                       seed=None,
                                                                                                       dtype=tf.float32))
        
         weights['fc_weights1'] = fc_weight1          
         fc1 = tf.matmul(activations,fc_weight1) 
         biases1 = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[64],name='fc_b1'))
         bias = tf.nn.bias_add(fc1,biases1)        
         fc_layer1 = tf.nn.relu(bias,name='ReLU_fc_1')
         weight_summaries(weights['fc_weights1'])
         
         
    with tf.name_scope('dropout'):
        
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        fc_layer1_dropped = tf.nn.dropout(fc_layer1, keep_prob)

    with tf.name_scope('fc_2'):
         fc_weight2 = tf.get_variable(name = 'fc_weights2',shape=[64,10],initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                                       mode='FAN_IN',
                                                                                                       uniform=False,
                                                                                                       seed=None,
                                                                                                       dtype=tf.float32))
        
         weights['fc_weights2'] = fc_weight2         
         fc2 = tf.matmul(fc_layer1_dropped,fc_weight2) 
         biases2 = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[10],name='fc_b2'))
         bias = tf.nn.bias_add(fc2,biases2)        
         fc_layer2 = tf.nn.relu(bias,name='ReLU_fc_2')
         weight_summaries(weights['fc_weights2'])
    
    #cross_entropy loss
    with tf.name_scope('cross_entropy_loss'):
         regularization = WEIGHT_DECAY * (tf.nn.l2_loss(weights['conv_weights1'])+\
                                             tf.nn.l2_loss(weights['conv_weights2'])+\
                                             tf.nn.l2_loss(weights['fc_weights1'])+\
                                             tf.nn.l2_loss(weights['fc_weights2']))         
         cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=_Ys,
                                                            logits = fc_layer2)+regularization)
                                                            
            
    #prediction    
    with tf.name_scope('y_Predicted'):
        _y_pred = tf.cast(tf.argmax(fc_layer2,1),dtype=tf.float32)     
    
    return _y_pred,cross_entropy,_Ys   
    
def weight_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('Weight_summary'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

def optimize(iterations,IMG_FLAT=IMG_FLAT,\
                            IMAGE_DIM=IMAGE_DIM,\
                            IMAGE_DEPTH=IMAGE_DEPTH,\
                            NUM_CLASSES=NUM_CLASSES,_train_count=1,_test_count=1):
         
         with tf.Graph().as_default():
             
             #to feed the network
             with tf.name_scope('Image'):
                 _Xs_images = tf.placeholder(tf.float32,shape=[None,IMG_FLAT],name='images')
                 
             with tf.name_scope('Label'):    
                 _Ys_labels = tf.placeholder(tf.int32,shape=[None],name='labels')
             
             with tf.name_scope('Dropout_probability'):
                 keep_prob = tf.placeholder(tf.float32)    

             with tf.name_scope('Model'):
                 #input the image and get the softmax output 
                 _y_pred,cross_entropy,_Ys = model(_Xs_images,_Ys_labels,keep_prob)            
         
             #optimization   
             with tf.name_scope('Optimization'):                                                   
                 train_net = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) 
                 
             with tf.name_scope('Accuracy'):
                 _y = tf.cast(tf.argmax(_Ys,1),dtype=tf.float32)
                 with tf.name_scope('Correct_prediction'):
                     #finding the accuracy         
                     correct_prediction = tf.equal(_y_pred,_y)
                 with tf.name_scope('accuracy'):
                     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
         
             tf.summary.scalar('accuracy', accuracy)
             # Create a summary to monitor cost tensor
             tf.summary.scalar("Loss", cross_entropy)
             # Create a summary to monitor accuracy tensor
             _op_summary =tf.summary.merge_all()
         
         
             ### SAVE PARAMETERS
             saver = tf.train.Saver()
             save_dir = save_path #directory name

         
             with tf.Session() as sess:
                 _train_summery_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
                 _test_summery_writer = tf.summary.FileWriter(log_dir + '/test', sess.graph)
                 
                 sess.run(tf.initialize_all_variables())
             
             
                 start_time = time.time()
                 for i in range(iterations):
                     
                     images,labels = get_data(batch=i%(NUM_BATCHES),isTraining=True)
                     
                     for j in range(images.shape[0] / BATCH_SIZE + 1):
                                                 
                         #preparing inputs
                         _trainXs,_trainYs = images[j*BATCH_SIZE:(j+1)*BATCH_SIZE,:],labels[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
                         feed_dict={_Xs_images:_trainXs,_Ys_labels:_trainYs,keep_prob:0.5}
                         #training/backprop
                         _train_summary,_,loss= sess.run([_op_summary,train_net,cross_entropy],feed_dict) 
                         _train_summery_writer.add_summary(_train_summary, _train_count)
                         _train_count += 1
                     #to evaluate on test set    
                     if i%NUM_BATCHES-1==0 and i>1:
                         _test_accuracy = 0.0
                         images,labels = get_data(batch=i%(NUM_BATCHES),isTraining=True)
                         for j in range(images.shape[0] / BATCH_SIZE + 1):
                            #preparing inputs
                            _trainXs,_trainYs = images[j*BATCH_SIZE:(j+1)*BATCH_SIZE,:],labels[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
                            feed_dict={_Xs_images:_trainXs,_Ys_labels:_trainYs,keep_prob:0.5}
                            #training/backprop
                            _test_summary,_mini_batch_acc,loss= sess.run([_op_summary,accuracy,cross_entropy],feed_dict) 
                            _test_accuracy +=  _mini_batch_acc / float(images.shape[0] / BATCH_SIZE + 1)                          
                            _test_summery_writer.add_summary(_test_summary, _test_count)
                            _test_count += 1
                         msg = "Accuracy on Test-Set After Epoch {0} : {1:.1%}"
                         print(msg.format(i/NUM_BATCHES,_test_accuracy))
                         
                 _train_summery_writer.close()       
                 _test_summery_writer.close()
                 
                 saver.save(sess = sess,save_path=save_dir)
                 print("Model stored in file: %s" % save_dir)  
                 #to moniter the time                 
                 end_time = time.time()
                 time_dif = end_time - start_time
                 print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


    
    
def inference():
    """
    predict on a test set
    """
    #to feed the network
    _Xs_images = tf.placeholder(tf.float32,shape=[None,IMG_FLAT],name='images')
    _Xs = tf.reshape(_Xs_images, shape=[-1, IMAGE_DIM,IMAGE_DIM,IMAGE_DEPTH])
    _Ys_labels = tf.placeholder(tf.int32,shape=[None],name='labels')
    _Ys = tf.one_hot(_Ys_labels,depth=NUM_CLASSES)
    #input the image and get the softmax output         
    fc_layer2 = model(_Xs)         
    
    # predicted output and actual output
    _y_pred = tf.cast(tf.argmax(fc_layer2,1),dtype=tf.float32)
    _y = tf.cast(tf.argmax(_Ys,1),dtype=tf.float32)
    #finding the accuracy         
    correct_prediction = tf.equal(_y_pred,_y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
         
    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        saver = tf.train.import_meta_graph('/home/jay/Deep_Structures/TF/my_test_model.meta')
        saver.restore(session,'/home/jay/Deep_Structures/TF/my_test_model')
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        print all_vars
        #session.run(tf.initialize_all_variables())
        #ckpt = tf.train.get_checkpoint_state(os.path.dirname('/home/jay/Deep Network Structures/Tensorflow/TrainedModels/'))
        #if ckpt and ckpt.model_checkpoint_path:
        #    tf.train.Saver.restore(session, ckpt.model_checkpoint_path)
        _batch_acc = []
        images,labels = get_data(isTraining=False)
        for j in range(images.shape[0] / BATCH_SIZE + 1):
        
            _trainXs = images[j*BATCH_SIZE:(j+1)*BATCH_SIZE,:]
            _trainYs= labels[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
                    
            feed_dict={_Xs_images:_trainXs,_Ys_labels:_trainYs}
            _miniAcc = session.run(accuracy,feed_dict)
            _batch_acc.append(_miniAcc)        
        msg = "Accuracy on Test-Set: {0:.1%}"
        print(msg.format(sum(_batch_acc)/float(len(_batch_acc))))
    
def get_data(batch=0,isTraining=True):

    ROOT_PATH = "/home/jay/Deep_Structures/cifar-10-python"
    #for training and testing separately
    if isTraining:
        path = os.path.join(ROOT_PATH,"train_batches/")
    else:
        path = os.path.join(ROOT_PATH,"test_batches/")
    file_names = [f for f in os.listdir(path)]
    batch = unpickle(os.path.join(path,file_names[batch]))
    images,labels = np.array(batch['data'],np.float32),batch['labels']     
    #feature scaling the data
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
                                