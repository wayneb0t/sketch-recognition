import os
import numpy as np
import PIL
from PIL import Image
from scipy.misc import imread, imresize, toimage

def get_data(num_classes=250, per_class=80, res=128):
    """
    Args:
        per_class: the number of images to read from each class
        res: the resolution of the output arrays (N x res x res)
        thresh: thresholding for the image
    
    """
    root_dir = "data/png{}/".format("" if res is None else res)
    
    per_class = min(80, per_class)
    
    labels = []
    
    X = np.zeros((num_classes * per_class, res, res, 1), dtype=np.float32)
    y = np.repeat(np.arange(num_classes), per_class)
    
    index = 0
    classes = 0
    for node in sorted(os.listdir(root_dir)):
        if os.path.isfile(root_dir + node):
            continue
        
        labels.append(node)
        label_path = root_dir + node + "/"
        num_images = 0
        for im_file in os.listdir(label_path):
            im_data = load_image(label_path + im_file)
            X[index] = im_data.reshape(res, res, 1)
            index += 1
            num_images += 1
            
            if num_images == per_class:
                break
                
        classes += 1
        if classes == num_classes:
            break

    return X, y, labels

def load_image(path):
    im_data = imread(path, mode='L')
    return im_data

def resize_images(res=128):
    root_dir = "data/png/"
    new_dir = "data/png{}/".format(res)
    try:
        os.mkdir(new_dir, 0755)
    except:
        pass
    
    for node in sorted(os.listdir(root_dir)):
        if os.path.isfile(root_dir + node):
            continue
            
        print "Resizing {}".format(node)
        
        label_path = root_dir + node + "/"
        new_path = new_dir + node + "/"
        
        try:
            os.mkdir(new_path, 0755)
        except:
            pass
        
        for im_file in os.listdir(label_path):
            im_path = label_path + im_file
            new_im_path = new_path + im_file
            
            img = Image.open(im_path)
            img = img.resize((res, res), PIL.Image.BILINEAR)
            img.save(new_im_path)
            img.close()