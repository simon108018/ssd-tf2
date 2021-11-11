from tensorflow.keras.layers import (Activation, Concatenate, Conv2D, Flatten,
                                     Input, Reshape)
from tensorflow.keras.models import Model

from nets.ssd_layers import Normalize, PriorBox
from nets.resnet import Backbone


def SSD300(input_shape, num_classes=21,anchors_size=[30,60,111,162,213,264,315], backbone_name='ResNet50'):
    #---------------------------------#
    #   典型的输入大小为[300,300,3]
    #---------------------------------#
    input_tensor = Input(shape=input_shape)
    
    # net变量里面包含了整个SSD的结构，通过层名可以找到对应的特征层
    net = Backbone(input_tensor, name=backbone_name)

    #-----------------------将提取到的主干特征进行处理---------------------------#
    # 对stage2_b的通道进行l2标准化处理
    # 38,38,512
    net['stage2_b_norm'] = Normalize(20, name='stage2_b_norm')(net['stage2_b'])
    num_priors = 4
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['stage2_b_norm_mbox_loc'] = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same', name='stage2_b_norm_mbox_loc')(net['stage2_b_norm'])
    net['stage2_b_norm_mbox_loc_flat'] = Flatten(name='stage2_b_norm_mbox_loc_flat')(net['stage2_b_norm_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['stage2_b_norm_mbox_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='stage2_b_norm_mbox_conf')(net['stage2_b_norm'])
    net['stage2_b_norm_mbox_conf_flat'] = Flatten(name='stage2_b_norm_mbox_conf_flat')(net['stage2_b_norm_mbox_conf'])

    priorbox = PriorBox(input_shape, anchors_size[0], max_size=anchors_size[1], aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='stage2_b_norm_mbox_priorbox')
    net['stage2_b_norm_mbox_priorbox'] = priorbox(net['stage2_b_norm'])
    
    # 对stage3_b层进行处理
    # 19,19,1024
    num_priors = 6
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['stage3_b_mbox_loc'] = Conv2D(num_priors * 4, kernel_size=(3,3),padding='same',name='stage3_b_mbox_loc')(net['stage3_b'])
    net['stage3_b_mbox_loc_flat'] = Flatten(name='stage3_b_mbox_loc_flat')(net['stage3_b_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['stage3_b_mbox_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3,3),padding='same',name='stage3_b_mbox_conf')(net['stage3_b'])
    net['stage3_b_mbox_conf_flat'] = Flatten(name='stage3_b_mbox_conf_flat')(net['stage3_b_mbox_conf'])

    priorbox = PriorBox(input_shape, anchors_size[1], max_size=anchors_size[2], aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='stage3_b_mbox_priorbox')
    net['stage3_b_mbox_priorbox'] = priorbox(net['stage3_b'])

    # 对stage4_b进行处理
    # 10,10,512
    num_priors = 6
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    x = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same',name='stage4_b_mbox_loc')(net['stage4_b'])
    net['stage4_b_mbox_loc'] = x
    net['stage4_b_mbox_loc_flat'] = Flatten(name='stage4_b_mbox_loc_flat')(net['stage4_b_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    x = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='stage4_b_mbox_conf')(net['stage4_b'])
    net['stage4_b_mbox_conf'] = x
    net['stage4_b_mbox_conf_flat'] = Flatten(name='stage4_b_mbox_conf_flat')(net['stage4_b_mbox_conf'])

    priorbox = PriorBox(input_shape, anchors_size[2], max_size=anchors_size[3], aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='stage4_b_mbox_priorbox')
    net['stage4_b_mbox_priorbox'] = priorbox(net['stage4_b'])

    # 对stage5_b进行处理
    # 5,5,256
    num_priors = 6
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    x = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same',name='stage5_b_mbox_loc')(net['stage5_b'])
    net['stage5_b_mbox_loc'] = x
    net['stage5_b_mbox_loc_flat'] = Flatten(name='stage5_b_mbox_loc_flat')(net['stage5_b_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    x = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='stage5_b_mbox_conf')(net['stage5_b'])
    net['stage5_b_mbox_conf'] = x
    net['stage5_b_mbox_conf_flat'] = Flatten(name='stage5_b_mbox_conf_flat')(net['stage5_b_mbox_conf'])

    priorbox = PriorBox(input_shape, anchors_size[3], max_size=anchors_size[4], aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='stage5_b_mbox_priorbox')
    net['stage5_b_mbox_priorbox'] = priorbox(net['stage5_b'])

    # 对stage6_b进行处理
    # 3,3,256
    num_priors = 4
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    x = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same',name='stage6_b_mbox_loc')(net['stage6_b'])
    net['stage6_b_mbox_loc'] = x
    net['stage6_b_mbox_loc_flat'] = Flatten(name='stage6_b_mbox_loc_flat')(net['stage6_b_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    x = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='stage6_b_mbox_conf')(net['stage6_b'])
    net['stage6_b_mbox_conf'] = x
    net['stage6_b_mbox_conf_flat'] = Flatten(name='stage6_b_mbox_conf_flat')(net['stage6_b_mbox_conf'])

    priorbox = PriorBox(input_shape, anchors_size[4], max_size=anchors_size[5], aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='stage6_b_mbox_priorbox')
    net['stage6_b_mbox_priorbox'] = priorbox(net['stage6_b'])

    # 对stage7_b进行处理
    # 1,1,256
    num_priors = 4
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    x = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same',name='stage7_b_mbox_loc')(net['stage7_b'])
    net['stage7_b_mbox_loc'] = x
    net['stage7_b_mbox_loc_flat'] = Flatten(name='stage7_b_mbox_loc_flat')(net['stage7_b_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    x = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='stage7_b_mbox_conf')(net['stage7_b'])
    net['stage7_b_mbox_conf'] = x
    net['stage7_b_mbox_conf_flat'] = Flatten(name='stage7_b_mbox_conf_flat')(net['stage7_b_mbox_conf'])
    
    priorbox = PriorBox(input_shape, anchors_size[5], max_size=anchors_size[6], aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='stage7_b_mbox_priorbox')

    net['stage7_b_mbox_priorbox'] = priorbox(net['stage7_b'])

    # 将所有结果进行堆叠
    net['mbox_loc'] = Concatenate(axis=1, name='mbox_loc')([net['stage2_b_norm_mbox_loc_flat'],
                             net['stage3_b_mbox_loc_flat'],
                             net['stage4_b_mbox_loc_flat'],
                             net['stage5_b_mbox_loc_flat'],
                             net['stage6_b_mbox_loc_flat'],
                             net['stage7_b_mbox_loc_flat']])
                            
    net['mbox_conf'] = Concatenate(axis=1, name='mbox_conf')([net['stage2_b_norm_mbox_conf_flat'],
                              net['stage3_b_mbox_conf_flat'],
                              net['stage4_b_mbox_conf_flat'],
                              net['stage5_b_mbox_conf_flat'],
                              net['stage6_b_mbox_conf_flat'],
                              net['stage7_b_mbox_conf_flat']])
                             
    net['mbox_priorbox'] = Concatenate(axis=1, name='mbox_priorbox')([net['stage2_b_norm_mbox_priorbox'],
                                  net['stage3_b_mbox_priorbox'],
                                  net['stage4_b_mbox_priorbox'],
                                  net['stage5_b_mbox_priorbox'],
                                  net['stage6_b_mbox_priorbox'],
                                  net['stage7_b_mbox_priorbox']])
                                  
    # 8732,4
    net['mbox_loc'] = Reshape((-1, 4),name='mbox_loc_final')(net['mbox_loc'])
    # 8732,21
    net['mbox_conf'] = Reshape((-1, num_classes),name='mbox_conf_logits')(net['mbox_conf'])
    # 8732,8
    net['mbox_conf'] = Activation('softmax', name='mbox_conf_final')(net['mbox_conf'])
    # 8732,33
    net['predictions'] = Concatenate(axis=2, name='predictions')([net['mbox_loc'],
                               net['mbox_conf'],
                               net['mbox_priorbox']])
                               
    model = Model(net['input'], net['predictions'])
    return model
