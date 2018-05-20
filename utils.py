import re
import numpy as np
from matplotlib import pyplot as plt
from skimage import transform
from keras.preprocessing.image import array_to_img, img_to_array, load_img


# Filename manupulation utils
def get_img_ID(fn):
    matched = re.search('.*/img(.*).jpg', fn)
    if matched:
        return matched.group(1)
    else:
        return ''


# Image manipulation utils
def resize(img, matrix_size_new):
    order = 0   # Nearest interpolation
    
    return np.stack( [ transform.resize(img[i,:], matrix_size_new, order=order, preserve_range=True, mode='constant') \
                       for i in range(img.shape[0]) ], \
                     axis = 0)


# Metrics utils
def calc_dice(m1, m2):
    return 2*((m1==1) & (m2==1)).sum()/((m1==1).sum() + (m2==1).sum() + 1e-6)


# I/O utils
def load_data(fn, flag_mask=False):
    img = img_to_array(load_img(fn), data_format='channels_first')
    
    if flag_mask:
        # Only take one channel
        img = img[0,:,:]
        # Threshold
        img[img < 128] = 0
        img[img >= 128] = 1
        
    return img
    
    
def save_data(img, fn, flag_mask=False):
    if flag_mask:
        img = np.tile(img[np.newaxis,:,:], (3,1,1)) * 255
    tmp = array_to_img(img)
    tmp.save(fn)
    
    return
    

# Plotting utils
def plots(ims, figsize=(12,6), rows=1, scale=None, interp=False, titles=None):
    
    if scale != None:
        lo, hi = scale
        ims = (ims - lo)/(hi - lo) * 255
        
    if(ims.ndim == 2):
        ims = np.tile(ims, (1,1,1));
    
    # move color to the last channel
    ims = np.transpose(ims, [1,2,0])
    
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if(ims.shape[-1] != 3):
            ims = np.tile(ims, (1,1,3));
            
    #print(ims.shape)
    f = plt.figure(figsize=figsize)
    plt.imshow(ims, interpolation=None if interp else 'none')