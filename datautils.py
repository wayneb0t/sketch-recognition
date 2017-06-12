import os
import numpy as np
import PIL
from PIL import Image
from scipy.misc import imread, imresize, toimage

def get_data(num_classes=250, res=128, flip=True, color_invert=True, center=False):
    """
    Generates the datasets with 128 (or 64) training examples, 8 validation examples,
    and 8 testing examples per class.
    
    Args:
        num_classes: the number of classes to load (for smaller datasets)
        res: the resolution of the output arrays (N x res x res)
        flip: whether or not to generate additional training examples by horizontally
              flipping the provided images
        color_invert: whether or not to invert B&W values
    """
    root_dir = "data/png{}/".format("" if res is None else res)
    
    num_train = 96 if flip else 48
    num_val = 16
    num_test = 16
    
    labels = []
    
    X_train = np.zeros((num_classes * num_train, res, res, 1), dtype=np.float32)
    y_train = np.repeat(np.arange(num_classes), num_train)
    
    X_val = np.zeros((num_classes * num_val, res, res, 1), dtype=np.float32)
    y_val = np.repeat(np.arange(num_classes), num_val)
    
    X_test = np.zeros((num_classes * num_test, res, res, 1), dtype=np.float32)
    y_test = np.repeat(np.arange(num_classes), num_test)
    
    classes = 0
    train_index = 0
    val_index = 0
    test_index = 0
    
    for node in sorted(os.listdir(root_dir)):
        if os.path.isfile(root_dir + node):
            continue
        
        labels.append(node)
        label_path = root_dir + node + "/"
        
        num_images = 0
        for im_file in sorted(os.listdir(label_path)):
            im_data = load_image(label_path + im_file).reshape(res, res, 1)
            
            if color_invert:
                im_data = -1 * im_data + 255
            
            if num_images < num_train:
                X_train[train_index] = im_data
                train_index += 1
                
                if flip:
                    X_train[train_index] = np.flip(im_data, axis=1)
                    train_index += 1
                    num_images += 1
                    
            elif num_images < num_train + num_val:
                X_val[val_index] = im_data
                val_index += 1
            else:
                X_test[test_index] = im_data
                test_index += 1
                
            num_images += 1
                
        classes += 1
        if classes == num_classes:
            break

    if center:
        X_train -= np.mean(X_train, axis=0)
        X_val -= np.mean(X_val, axis=0)
        X_test -= np.mean(X_test, axis=0)
    return X_train, y_train, X_val, y_val, X_test, y_test, labels

def get_cross_val_data(num_classes=250, res=128, flip=True, color_invert=True, center=False):
    """
    Generates the dataset splits for cross validation
    
    Args:
        num_classes: the number of classes to load (for smaller datasets)
        res: the resolution of the output arrays (N x res x res)
        flip: whether or not to generate additional training examples by horizontally
              flipping the provided images
        color_invert: whether or not to invert B&W values
    """
    root_dir = "data/png{}/".format("" if res is None else res)
    
    split1 = 27
    split2 = 27
    split3 = 26
    
    labels = []

    X1 = np.zeros((num_classes * split1, res, res, 1), dtype=np.float32)
    y1 = np.repeat(np.arange(num_classes), split1)
    X2 = np.zeros((num_classes * split2, res, res, 1), dtype=np.float32)
    y2 = np.repeat(np.arange(num_classes), split2)
    X3 = np.zeros((num_classes * split3, res, res, 1), dtype=np.float32)
    y3 = np.repeat(np.arange(num_classes), split3)
    
    classes = 0
    index1 = 0
    index2 = 0
    index3 = 0
    
    for node in sorted(os.listdir(root_dir)):
        if os.path.isfile(root_dir + node):
            continue
        
        labels.append(node)
        label_path = root_dir + node + "/"
        
        num_images = 0
        for im_file in sorted(os.listdir(label_path)):
            im_data = load_image(label_path + im_file).reshape(res, res, 1)
            
            if color_invert:
                im_data = -1 * im_data + 255
            
            if num_images < split1:
                X1[index1] = im_data
                index1 += 1
            elif num_images < split1 + split2:
                X2[index2] = im_data
                index2 += 1
            else:
                X3[index3] = im_data
                index3 += 1
                
            num_images += 1
                
        classes += 1
        if classes == num_classes:
            break
            
    X_train1 = np.concatenate((X1, X2), axis=0)
    y_train1 = np.concatenate((y1, y2))
    X_test1 = X3
    y_test1 = y3
    
    X_train2 = np.concatenate((X1, X3), axis=0)
    y_train2 = np.concatenate((y1, y3))
    X_test2 = X2
    y_test2 = y2
    
    X_train3 = np.concatenate((X2, X3), axis=0)
    y_train3 = np.concatenate((y2, y3))
    X_test3 = X1
    y_test3 = y1
    
    if flip:
        X_train1 = np.concatenate((X_train1, np.flip(X_train1, axis=2)), axis=0)
        y_train1 = np.concatenate((y_train1, y_train1))
        X_train2 = np.concatenate((X_train2, np.flip(X_train2, axis=2)), axis=0)
        y_train2 = np.concatenate((y_train2, y_train2))
        X_train3 = np.concatenate((X_train3, np.flip(X_train3, axis=2)), axis=0)
        y_train3 = np.concatenate((y_train3, y_train3))

    if center:
        X_train1 -= np.mean(X_train1, axis=0)
        X_test1 -= np.mean(X_test1, axis=0)
        X_train2 -= np.mean(X_train2, axis=0)
        X_test2 -= np.mean(X_test2, axis=0)
        X_train3 -= np.mean(X_train3, axis=0)
        X_test3 -= np.mean(X_test3, axis=0)
    set1 = (X_train1, y_train1, X_test1, y_test1)
    set2 = (X_train2, y_train2, X_test2, y_test2)
    set3 = (X_train3, y_train3, X_test3, y_test3)
    
    return set1, set2, set3, labels

def load_image(path):
    im_data = imread(path, mode='L')
    return im_data

def resize_images(res=128):
    root_dir = "data/png/"
    new_dir = "data/png{}/".format(res)
    try:
        os.mkdir(new_dir, 755)
    except:
        pass
    
    for node in sorted(os.listdir(root_dir)):
        if os.path.isfile(root_dir + node):
            continue
            
        print ("Resizing {}".format(node))
        
        label_path = root_dir + node + "/"
        new_path = new_dir + node + "/"
        
        try:
            os.mkdir(new_path, 755)
        except:
            pass
        
        for im_file in os.listdir(label_path):
            im_path = label_path + im_file
            new_im_path = new_path + im_file
            
            img = Image.open(im_path)
            img = img.resize((res, res), PIL.Image.BILINEAR)
            img.save(new_im_path)
            img.close()