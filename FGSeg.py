import sys, getopt
import os

import numpy as np
import glob

from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

from utils import *
from unet import generate_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def main(argv):
    
    ## ============  Determine input argumentts =============== ##
    filename_in = ''
    filename_out = ''
    
    try:
        opts, args = getopt.getopt(argv,"hi:o:",[])
    except getopt.GetoptError:
        print('FGSeg.py -i <filename_in> -o <filename_out>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('FGSeg.py -i <filename_in> -o <filename_out>')
            sys.exit()
        elif opt == '-i':
            filename_in = arg
        elif opt == '-o':
            filename_out = arg
            
    if not filename_in or not filename_out:
        print('Empty name(s).')
        sys.exit(2)


    ## =================== Parameter setting ===================== ##
    matrix_size = (320, 400)
    num_classes = 2
    num_channel = 3

    # model weights
    fn_model = os.path.dirname(__file__) + '/models/weights_optimal.h5'
    
    
    
    ## ======================= Prepare data ====================== ##
    img = load_data(filename_in)
    
    matrix_size_raw = img.shape[1:]
    
    # Resize image
    img_resized = resize(img, matrix_size)
    
    
    
    ## ======================= Prepare Model ===================== ##
    model = generate_model(num_classes, num_channel, input_size=matrix_size, output_size=matrix_size)

    model.load_weights(fn_model)
    
    
    
    ## ========================= Run Model ======================= ##
    pred = model.predict(img_resized[np.newaxis,:], verbose=1)[0,:]
    pred_classes = np.argmax(pred, axis=-1)
    
    mask_pred = pred_classes
    
    # resize result back to original size
    mask = resize(mask_pred[np.newaxis,:], matrix_size_raw)[0,:,:]
    
    
    
    ## ======================= Save Result ======================= ##
    save_data(mask, filename_out, flag_mask=True)
    
    
    print('Segmentation successful.')
    
    
    
if __name__ == "__main__":
    main(sys.argv[1:])
    