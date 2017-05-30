import os
import numpy as np
import PIL
from PIL import Image
from scipy.misc import imread, imresize, toimage


def get_data(num_classes=250, res=128, flip=True):
    """
    Generates the datasets with 128 (or 64) training examples, 8 validation examples,
    and 8 testing examples per class.
    
    Args:
        num_classes: the number of classes to load (for smaller datasets)
        res: the resolution of the output arrays (N x res x res)
        flip: whether or not to generate additional training examples by horizontally
              flipping the provided images
    """
    root_dir = "data/png{}/".format("" if res is None else res)
    
    num_train = 128 if flip else 64
    num_val = 8
    num_test = 8
    
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
        for im_file in os.listdir(label_path):
            im_data = load_image(label_path + im_file).reshape(res, res, 1)
            
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

    X_train -= np.mean(X_train, axis=0)
    X_val -= np.mean(X_val, axis=0)
    X_test -= np.mean(X_test, axis=0)
    return X_train, y_train, X_val, y_val, X_test, y_test, labels

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