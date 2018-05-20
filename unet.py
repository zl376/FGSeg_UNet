from keras import backend as K
from keras.layers import Activation
from keras.layers import Input
from keras.layers import BatchNormalization
#from keras.layers import Dropout
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Cropping2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Permute
from keras.layers.core import Reshape
from keras.layers.merge import concatenate
from keras.models import Model

K.set_image_dim_ordering('th')

def generate_model(num_classes, num_channel=3, input_size=(300, 400), output_size=(300, 400)) :
    
    # U-Net for binary segmentation
    
    inputs = Input((num_channel,) + input_size)

    conv1 = Conv2D(64, kernel_size=(3, 3), padding='same')(inputs)
    conv1 = BatchNormalization(axis=1)(PReLU()(conv1))
    conv1 = Conv2D(64, kernel_size=(3, 3), padding='same')(conv1)
    conv1 = BatchNormalization(axis=1)(PReLU()(conv1))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, kernel_size=(3, 3), padding='same')(pool1)
    conv2 = BatchNormalization(axis=1)(PReLU()(conv2))
    conv2 = Conv2D(128, kernel_size=(3, 3), padding='same')(conv2)
    conv2 = BatchNormalization(axis=1)(PReLU()(conv2))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)    
    
    conv3 = Conv2D(256, kernel_size=(3, 3), padding='same')(pool2)
    conv3 = BatchNormalization(axis=1)(PReLU()(conv3))
    conv3 = Conv2D(256, kernel_size=(3, 3), padding='same')(conv3)
    conv3 = BatchNormalization(axis=1)(PReLU()(conv3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  
    
    conv4 = Conv2D(512, kernel_size=(3, 3), padding='same')(pool3)
    conv4 = BatchNormalization(axis=1)(PReLU()(conv4))
    conv4 = Conv2D(512, kernel_size=(3, 3), padding='same')(conv4)
    conv4 = BatchNormalization(axis=1)(PReLU()(conv4))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4) 
    
    conv5 = Conv2D(1024, kernel_size=(3, 3), padding='same')(pool4)
    conv5 = BatchNormalization(axis=1)(PReLU()(conv5))
    conv5 = Conv2D(1024, kernel_size=(3, 3), padding='same')(conv5)
    conv5 = BatchNormalization(axis=1)(PReLU()(conv5))
    
    up6 = Conv2D(512, kernel_size=(2, 2), padding='same')(UpSampling2D(size=(2, 2))(conv5))
    concat6 = concatenate([conv4, up6], axis=1)
    conv6 = Conv2D(512, kernel_size=(3, 3), padding='same')(concat6)
    conv6 = BatchNormalization(axis=1)(PReLU()(conv6))   
    conv6 = Conv2D(512, kernel_size=(3, 3), padding='same')(conv6)
    conv6 = BatchNormalization(axis=1)(PReLU()(conv6))    
    
    up7 = Conv2D(256, kernel_size=(2, 2), padding='same')(UpSampling2D(size=(2, 2))(conv6))
    concat7 = concatenate([conv3, up7], axis=1)
    conv7 = Conv2D(256, kernel_size=(3, 3), padding='same')(concat7)
    conv7 = BatchNormalization(axis=1)(PReLU()(conv7))   
    conv7 = Conv2D(256, kernel_size=(3, 3), padding='same')(conv7)
    conv7 = BatchNormalization(axis=1)(PReLU()(conv7))    
    
    up8 = Conv2D(128, kernel_size=(2, 2), padding='same')(UpSampling2D(size=(2, 2))(conv7))
    concat8 = concatenate([conv2, up8], axis=1)
    conv8 = Conv2D(128, kernel_size=(3, 3), padding='same')(concat8)
    conv8 = BatchNormalization(axis=1)(PReLU()(conv8))   
    conv8 = Conv2D(128, kernel_size=(3, 3), padding='same')(conv8)
    conv8 = BatchNormalization(axis=1)(PReLU()(conv8))       
    
    up9 = Conv2D(64, kernel_size=(2, 2), padding='same')(UpSampling2D(size=(2, 2))(conv8))
    concat9 = concatenate([conv1, up9], axis=1)
    conv9 = Conv2D(64, kernel_size=(3, 3), padding='same')(concat9)
    conv9 = BatchNormalization(axis=1)(PReLU()(conv9))   
    conv9 = Conv2D(64, kernel_size=(3, 3), padding='same')(conv9)
    conv9 = BatchNormalization(axis=1)(PReLU()(conv9))       
    
    pred = Conv2D(num_classes, kernel_size=(1, 1))(conv9)
    pred = PReLU()(pred)
    pred = Reshape((num_classes, output_size[0]*output_size[1]))(pred)
    pred = Permute((2, 1))(pred)
    pred = Activation('softmax')(pred)
    pred = Reshape(output_size + (num_classes,))(pred)

    model = Model(inputs=inputs, outputs=pred)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['categorical_accuracy'])
    
    return model