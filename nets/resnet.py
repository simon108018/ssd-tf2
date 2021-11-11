# -------------------------------------------------------------#
#   ResNet的网络部分
# -------------------------------------------------------------#
from typing import Optional, Dict, Any, Union, Tuple

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (Layer, Activation, BatchNormalization, Conv2D, GlobalAvgPool2D,
                                     Dense, Conv2DTranspose, Add, MaxPooling2D, ZeroPadding2D)

from tensorflow.keras.regularizers import l2
from tensorflow.keras import initializers


class apply_ltrb(Layer):
    def __init__(self, name=None, **kwargs):
        super(apply_ltrb, self).__init__(name=name, **kwargs)

    def get_config(self):
        config = super(apply_ltrb, self).get_config()
        return config

    def call(self, pred_ltrb):
        '''
        pred_ltrb 上的4個value分別是(x1, y1, x2, y2)表示以每個cell為中心，預測出來的框架左上角與右下角的相對距離
        ltrb(left-up-right-bottom)
        此函數將預測出來的相對位置換算成絕對位置
        下面是一個框，在cell(cx,cy)取得相對距離(x1,y1,x2,y2)後，換算成絕對位置(cx-x1,cy-y1,cx+x2,cy+y2)
        (cx-x1,cy-y1)
          ----------------------------------
          |          ↑                     |
          |          |                     |
          |          |y1                   |
          |          |                     |
          |←------(cx,cy)-----------------→|
          |   x1     |          x2         |
          |          |                     |
          |          |                     |
          |          |y2                   |
          |          |                     |
          |          |                     |
          |          ↓                     |
          ----------------------------------(cx+x2,cy+y2)
        '''
        b, w, h, c = tf.shape(pred_ltrb)[0], tf.shape(pred_ltrb)[1], tf.shape(pred_ltrb)[2], tf.shape(pred_ltrb)[3]
        ct = tf.cast(tf.transpose(tf.meshgrid(tf.range(0, w), tf.range(0, h))), tf.float32)
        # locations : w*h*2 這2個 value包含 cx=ct[0], cy=ct[1]
        locations = tf.concat((ct - pred_ltrb[:, :, :, :2], ct + pred_ltrb[:, :, :, 2:]), axis=-1)
        return locations

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def BasicBlock(input_tensor, kernel_size, filters, stage, block, strides=(1, 1)):

    filters1, filters2 = filters

    conv_name_base = 'conv' + str(stage) + '_' + block
    bn_name_base = 'bn' + str(stage) + '_' + block

    x = Conv2D(filters1, kernel_size, strides=strides, padding='same',
               name=conv_name_base + '_0', use_bias=False)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '_0', momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '_1', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '_1', momentum=0.9, epsilon=1e-5)(x)

    if strides != (1, 1):
        shortcut = Conv2D(filters2, (1, 1), strides=strides, padding='same',
                          name=conv_name_base + '_shortcut', use_bias=False)(input_tensor)
        shortcut = BatchNormalization(name=bn_name_base + '_shortcut', momentum=0.9, epsilon=1e-5)(shortcut)
    else:
        shortcut = input_tensor

    x = Add()([x, shortcut])
    x = Activation('relu', name='stage{}_{}'.format(stage, block))(x)
    return x

def ResNet18_model(image_input=tf.keras.Input(shape=(300, 300, 3))) -> tf.keras.Model:
    # 150,150,64
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv0', padding='same', use_bias=False)(image_input)
    x = BatchNormalization(name='bn', momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)

    # 150,150,64 -> 75,75,64
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # 75,75,64 -> 75,75,64
    x = BasicBlock(x, 3, [64, 64], stage=1, block='a', strides=(1, 1))
    x = BasicBlock(x, 3, [64, 64], stage=1, block='b', strides=(1, 1))

    # 75,75,64 -> 38,38,128
    x = BasicBlock(x, 3, [128, 128], stage=2, block='a', strides=(2, 2))
    x = BasicBlock(x, 3, [128, 128], stage=2, block='b', strides=(1, 1))

    # 38,38,128 -> 19,19,256
    x = BasicBlock(x, 3, [256, 256], stage=3, block='a', strides=(2, 2))
    x = BasicBlock(x, 3, [256, 256], stage=3, block='b', strides=(1, 1))

    # 19,19,256 -> 10,10,512
    x = BasicBlock(x, 3, [512, 512], stage=4, block='a', strides=(2, 2))
    x = BasicBlock(x, 3, [512, 512], stage=4, block='b', strides=(1, 1))
    x = GlobalAvgPool2D()(x)
    x = Dense(1000, name='fully_connected', activation='softmax', use_bias=False)(x)

    return tf.keras.models.Model(inputs=image_input, outputs=x)


def ResNet18(image_input=tf.keras.Input(shape=(300,300,3))):
    net = {}
    model = ResNet18_model(image_input)
    model.load_weights('./my_ResNet_18.h5')
    net['input'] = model.inputs
    # 75, 75, 64
    net['stage1_b'] = model.get_layer('stage1_b').output
    # 38, 38, 128
    net['stage2_b'] = model.get_layer('stage2_b').output
    # 19, 19, 256
    net['stage3_b'] = model.get_layer('stage3_b').output
    # 10, 10, 512
    net['stage4_b'] = model.get_layer('stage4_b').output

    return net

def ResNet50(image_input=tf.keras.Input(shape=(300,300,3))):
    net = {}
    model = tf.keras.applications.ResNet50(include_top=False, input_tensor=image_input)
    net['input'] = model.inputs
    #  75, 75,  256
    net['stage1_b'] = model.get_layer('conv2_block3_out').output
    #  38, 38,  512
    net['stage2_b'] = model.get_layer('conv3_block4_out').output
    #  19, 19, 1024
    net['stage3_b'] = model.get_layer('conv4_block6_out').output
    #  10, 10, 2048
    net['stage4_b'] = model.get_layer('conv5_block3_out').output
    return net


def Backbone(image_input=tf.keras.Input(shape=(300, 300, 3)), name='resnet18'):
    if name.lower() == 'resnet18':
        net = ResNet18(image_input)
    elif name.lower() == 'resnet50':
        net = ResNet50(image_input)
    # # 10, 10, 512 -> 5, 5, 256
    net['stage5_a'] = Conv2D(128, kernel_size=(1,1), activation='relu',
                                   padding='same',
                                   name='stage5_a')(net['stage4_b'])
    net['stage5_padding'] = ZeroPadding2D(padding=((1, 1), (1, 1)), name='stage5_padding')(net['stage5_a'])
    net['stage5_b'] = Conv2D(256, kernel_size=(3,3), strides=(2, 2),
                                   activation='relu', padding='valid',
                                   name='stage5_b')(net['stage5_padding'])
    # 5, 5, 256 -> 3, 3, 256
    net['stage6_a'] = Conv2D(128, kernel_size=(1,1), activation='relu',
                                   padding='same',
                                   name='stage6_a')(net['stage5_b'])
    net['stage6_b'] = Conv2D(256, kernel_size=(3,3), strides=(1, 1),
                                   activation='relu', padding='valid',
                                   name='stage6_b')(net['stage6_a'])
    # 3, 3, 256 -> 1, 1, 256
    net['stage7_a'] = Conv2D(128, kernel_size=(1,1), activation='relu',
                                   padding='same',
                                   name='stage7_a')(net['stage6_b'])
    net['stage7_b'] = Conv2D(256, kernel_size=(3,3), strides=(1, 1),
                                   activation='relu', padding='valid',
                                   name='stage7_b')(net['stage7_a'])

    return net


