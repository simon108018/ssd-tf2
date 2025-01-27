from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        TensorBoard)
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from nets.ssd import SSD300
from nets.ssd_training import (Generator, LossHistory, ModelCheckpoint,
                               MultiboxLoss)
from utils.anchors import get_anchors
from utils.utils import BBoxUtility


# 防止bug
def get_train_step_fn():
    @tf.function
    def train_step(imgs, multiloss, targets, net, optimizer):
        with tf.GradientTape() as tape:
            # 计算loss
            prediction = net(imgs, training=True)
            loss_value = multiloss(targets, prediction)

        grads = tape.gradient(loss_value, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        return loss_value
    return train_step

@tf.function
def val_step(imgs, multiloss, targets, net):
    # 计算loss
    prediction = net(imgs)
    loss_value = multiloss(targets, prediction)

    return loss_value

def fit_one_epoch(net, multiloss, optimizer, epoch, epoch_size, epoch_size_val, gen, genval, 
                Epoch, train_step):

    total_loss = 0
    val_loss = 0
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration>=epoch_size:
                break
            images, targets = batch[0], batch[1]
            targets = tf.convert_to_tensor(targets)
            
            loss_value = train_step(images, multiloss, targets, net, optimizer)
            total_loss += loss_value

            pbar.set_postfix(**{'total_loss'        : float(total_loss) / (iteration + 1), 
                                'lr'                : optimizer._decayed_lr(tf.float32).numpy()})
            pbar.update(1)

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration>=epoch_size_val:
                break
            images, targets = batch[0], batch[1]
            targets = tf.convert_to_tensor(targets)

            loss_value = val_step(images, multiloss, targets, net)
            val_loss = val_loss + loss_value

            pbar.set_postfix(**{'total_loss': float(val_loss)/ (iteration + 1)})
            pbar.update(1)

    logs = {'loss': total_loss.numpy()/(epoch_size+1), 'val_loss': val_loss.numpy()/(epoch_size_val+1)}
    loss_history.on_epoch_end([], logs)
    print('Finish Validation')
    print('\nEpoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    net.save_weights('logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.h5'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
      
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
#----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
if __name__ == "__main__":
    log_dir = "logs/"
    annotation_path = '2007_train.txt'
    #----------------------------------------------------#
    #   是否使用eager模式训练
    #----------------------------------------------------#
    eager = False
    #----------------------------------------------------#
    #   训练之前一定要修改NUM_CLASSES
    #   修改成所需要区分的类的个数+1。
    #----------------------------------------------------#
    NUM_CLASSES = 21
    #----------------------------------------------------#
    #   input_shape有两个选择。
    #   一个是(300, 300, 3)、一个是(512, 512, 3)。
    #   这里的SSD512不是原版的SSD512。
    #   原版的SSD512的比SSD300多一个预测层；
    #   修改起来比较麻烦，所以我只是修改了输入大小
    #   这样也可以用比较大的图片训练，对于小目标有好处
    #----------------------------------------------------#
    input_shape = [300, 300, 3]

    #----------------------------------------------------#
    #   可用于设定先验框的大小，默认的anchors_size
    #   是根据voc数据集设定的，大多数情况下都是通用的！
    #   如果想要检测小物体，可以修改anchors_size
    #   一般调小浅层先验框的大小就行了！因为浅层负责小物体检测！
    #   比如anchors_size = [21,45,99,153,207,261,315]
    #----------------------------------------------------#
    anchors_size = [30,60,111,162,213,264,315]
    priors = get_anchors((input_shape[0], input_shape[1]), anchors_size)
    bbox_util = BBoxUtility(NUM_CLASSES, priors)

    model = SSD300(input_shape, NUM_CLASSES, anchors_size)
    #------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #   训练自己的数据集时提示维度不匹配正常
    #   预测的东西都不一样了自然维度不匹配
    #------------------------------------------------------#
    model_path = 'model_data/ssd_weights.h5'
    model.load_weights(model_path, by_name=True, skip_mismatch=True)

    #-------------------------------------------------------------------------------#
    #   训练参数的设置
    #   logging表示tensorboard的保存地址
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    #-------------------------------------------------------------------------------#
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    loss_history = LossHistory(log_dir)

    multiloss = MultiboxLoss(NUM_CLASSES, neg_pos_ratio=3.0).compute_loss
    #----------------------------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    #   2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
    #   当前划分方式下，验证集和训练集的比例为1:9
    #----------------------------------------------------------------------#
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    for i in range(21):
        model.layers[i].trainable = False
    if True:
        Init_epoch          = 0
        Freeze_epoch        = 50
        Batch_size          = 16
        learning_rate_base  = 5e-4

        epoch_size          = num_train // Batch_size
        epoch_size_val      = num_val // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, Batch_size))
        if eager:
            generator = Generator(bbox_util, Batch_size, lines[:num_train], lines[num_train:],
                        (input_shape[0], input_shape[1]),NUM_CLASSES)

            gen = tf.data.Dataset.from_generator(partial(generator.generate, train = True), (tf.float32, tf.float32))
            gen_val = tf.data.Dataset.from_generator(partial(generator.generate, train = False), (tf.float32, tf.float32))

            gen = gen.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)
            gen_val = gen_val.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)

            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate_base,
                decay_steps=epoch_size,
                decay_rate=0.95,
                staircase=True
            )

            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            for epoch in range(Init_epoch, Freeze_epoch):
                fit_one_epoch(model, multiloss, optimizer, epoch, epoch_size, epoch_size_val, gen, gen_val, 
                            Freeze_epoch, get_train_step_fn())
            
        else:
            gen = Generator(bbox_util, Batch_size, lines[:num_train], lines[num_train:],
                        (input_shape[0], input_shape[1]),NUM_CLASSES)
            model.compile(optimizer=Adam(lr=learning_rate_base),loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=3.0).compute_loss)
            
            model.fit(gen.generate(True), 
                    steps_per_epoch=epoch_size,
                    validation_data=gen.generate(False),
                    validation_steps=epoch_size_val,
                    epochs=Freeze_epoch, 
                    initial_epoch=Init_epoch,
                    callbacks=[logging, checkpoint, reduce_lr, early_stopping, loss_history])

    for i in range(21):
        model.layers[i].trainable = True
    if True:
        Freeze_epoch        = 50
        Epoch               = 100
        Batch_size          = 8
        learning_rate_base  = 1e-4
        
        epoch_size          = num_train // Batch_size
        epoch_size_val      = num_val // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, Batch_size))
        if eager:

            generator =Generator(bbox_util, Batch_size, lines[:num_train], lines[num_train:],
                        (input_shape[0], input_shape[1]),NUM_CLASSES)

            gen = tf.data.Dataset.from_generator(partial(generator.generate, train = True), (tf.float32, tf.float32))
            gen_val = tf.data.Dataset.from_generator(partial(generator.generate, train = False), (tf.float32, tf.float32))

            gen = gen.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)
            gen_val = gen_val.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)

            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate_base,
                decay_steps=epoch_size,
                decay_rate=0.95,
                staircase=True
            )

            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, Batch_size))
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            
            for epoch in range(Freeze_epoch, Epoch):
                fit_one_epoch(model, multiloss, optimizer, epoch, epoch_size, epoch_size_val, gen, gen_val, 
                            Epoch, get_train_step_fn())

        else:
            gen = Generator(bbox_util, Batch_size, lines[:num_train], lines[num_train:],
                        (input_shape[0], input_shape[1]),NUM_CLASSES)
            model.compile(optimizer=Adam(lr=learning_rate_base),loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=3.0).compute_loss)

            model.fit(gen.generate(True), 
                    steps_per_epoch=epoch_size,
                    validation_data=gen.generate(False),
                    validation_steps=epoch_size_val,
                    epochs=Epoch, 
                    initial_epoch=Freeze_epoch,
                    callbacks=[logging, checkpoint, reduce_lr, early_stopping, loss_history])
