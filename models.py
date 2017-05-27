import tensorflow as tf
import numpy as np

def naive_model(X, y, num_classes=250):
    c1 = tf.layers.conv2d(X, 32, [7, 7], padding='SAME') # 128 x 128 x 32
    b1 = tf.layers.batch_normalization(c1)
    h1 = tf.nn.relu(b1)
    p1 = tf.layers.max_pooling2d(h1, [2, 2], [2, 2]) # 64 x 64 x 32
    
    c2 = tf.layers.conv2d(p1, 64, [5, 5], padding='SAME') # 64 x 64 x 64
    b2 = tf.layers.batch_normalization(c2)
    h2 = tf.nn.relu(b2)
    p2 = tf.layers.max_pooling2d(h2, [2, 2], [2, 2]) # 32 x 32 x 64
    
    c3 = tf.layers.conv2d(p2, 128, [3, 3], padding = 'SAME') # 32 x 32 x 128
    b3 = tf.layers.batch_normalization(c3)
    h3 = tf.nn.relu(b3)
    p3 = tf.layers.max_pooling2d(h3, [2, 2], [2, 2]) # 16 x 16 x 128
    
    #p4 = tf.layers.average_pooling2d(p3, [32, 32], [1, 1]) # 1 x 1 x 64
    
    p3_flat = tf.reshape(p3, [-1,32768])
    y_out = tf.layers.dense(p3_flat, num_classes)
    
    return y_out

def resnet(X, y, layer_depth=4, num_classes=250):
    # RESnet-ish
    reg = 1e-2
    l2_reg = tf.contrib.layers.l2_regularizer(reg)

    """
    Input: 128x128x1
    Output: 64x64x64
    """
    c0 = tf.layers.conv2d(X, 64, [7, 7], strides=[2, 2], padding='SAME', kernel_regularizer=l2_reg)
    c0 = tf.layers.batch_normalization(c0)
    match_dimensions = True
    for i in range(layer_depth):
        c1 = tf.layers.conv2d(c0, 64, [3, 3], padding='SAME', kernel_regularizer=l2_reg) #conv
        b1 = tf.layers.batch_normalization(c1) #bn
        h1 = tf.nn.relu(b1) #relu
        c2 = tf.layers.conv2d(h1, 64, [3, 3], padding='SAME', kernel_regularizer=l2_reg) #conv
        b2 = tf.layers.batch_normalization(c2) #bn
        r = c0 + b2
        c0 = tf.nn.relu(r)
    
    """
    Input: 64x64x64
    Output: 32x32x128
    """
    downsample = True
    for i in range(layer_depth):
        c1 = tf.layers.conv2d(c0, 128, [3, 3], 
                              strides=([2, 2] if downsample else [1, 1]),
                              padding='SAME',
                              kernel_regularizer=l2_reg)
        b1 = tf.layers.batch_normalization(c1) #bn
        h1 = tf.nn.relu(b1) #relu
        c2 = tf.layers.conv2d(h1, 128, [3, 3], padding='SAME', kernel_regularizer=l2_reg) #conv
        b2 = tf.layers.batch_normalization(c2) #bn
        if downsample:
            c0_proj = tf.layers.conv2d(c0, 128, [1, 1], padding='SAME', kernel_regularizer=l2_reg)
            c0_proj = tf.layers.average_pooling2d(c0_proj, (2, 2), (2, 2))
            r = c0_proj + b2
            downsample = False
        else:
            r = c0 + b2
        c0 = tf.nn.relu(r)

    """
    Input: 32x32x128
    Output: 16x16x256
    """
    downsample = True
    for i in range(layer_depth):
        c1 = tf.layers.conv2d(c0, 256, [3, 3], 
                              strides=([2, 2] if downsample else [1, 1]),
                              padding='SAME',
                              kernel_regularizer=l2_reg)
        b1 = tf.layers.batch_normalization(c1) #bn
        h1 = tf.nn.relu(b1) #relu
        c2 = tf.layers.conv2d(h1, 256, [3, 3], padding='SAME', kernel_regularizer=l2_reg) #conv
        b2 = tf.layers.batch_normalization(c2) #bn
        if downsample:
            c0_proj = tf.layers.conv2d(c0, 256, [1, 1], padding='SAME', kernel_regularizer=l2_reg)
            c0_proj = tf.layers.average_pooling2d(c0_proj, (2, 2), (2, 2))
            r = c0_proj + b2
            downsample = False
        else:
            r = c0 + b2
        c0 = tf.nn.relu(r)

    """
    Input: 16x16x256
    Output: 8x8x512
    """
    downsample = True
    for i in range(layer_depth):
        c1 = tf.layers.conv2d(c0, 512, [3, 3], 
                              strides=([2, 2] if downsample else [1, 1]),
                              padding='SAME',
                              kernel_regularizer=l2_reg)
        b1 = tf.layers.batch_normalization(c1) #bn
        h1 = tf.nn.relu(b1) #relu
        c2 = tf.layers.conv2d(h1, 512, [3, 3], padding='SAME', kernel_regularizer=l2_reg) #conv
        b2 = tf.layers.batch_normalization(c2) #bn
        if downsample:
            c0_proj = tf.layers.conv2d(c0, 512, [1, 1], padding='SAME', kernel_regularizer=l2_reg)
            c0_proj = tf.layers.average_pooling2d(c0_proj, (2, 2), (2, 2))
            r = c0_proj + b2
            downsample = False
        else:
            r = c0 + b2
        c0 = tf.nn.relu(r)
    
    p1 = tf.layers.average_pooling2d(c0, (8, 8), (1,1))
    p1_flat = tf.reshape(p1, [-1, 512])
    y_out = tf.layers.dense(p1_flat, num_classes, kernel_regularizer=l2_reg)
    
    return y_out 

def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training
    
    # counter 
    iter_cnt = 0
    for e in range(epochs):
                # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
                        # generate indicies for the batch
            start_idx = (i*batch_size)%X_train.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:], y: yd[idx] }
                        
                        
            # get batch size
            actual_batch_size = yd[i:i+batch_size].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables,feed_dict=feed_dict)
            
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                                                              .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
                iter_cnt += 1
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
                              .format(total_loss,total_correct,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss,total_correct
