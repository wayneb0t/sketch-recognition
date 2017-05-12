import os
import numpy as np
from scipy.misc import imread, imresize, toimage

def get_data(num_classes=250, per_class=80, res=200, thresh=225):
    """
    Args:
        per_class: the number of images to read from each class
        res: the resolution of the output arrays (N x res x res)
        thresh: thresholding for the image
    
    """
    root_dir = "data/png/"
    
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
            im_data = imresize(imread(label_path + im_file, mode='1'), (res, res)).reshape(res, res, 1)
            #im_data[im_data < thresh] = 0
            #im_data[im_data >= thresh] = 255
            X[index] = im_data
            index += 1
            num_images += 1
            
            if num_images == per_class:
                break
                
        classes += 1
        if classes == num_classes:
            break

    return X, y, labels
