import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate
from keras.applications. resnet50 import ResNet50
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.layers import Activation, Dense,BatchNormalization
import sys, os
sys.path.append(os.path.abspath(os.path.join('../', 'models/')))


def build_resnet50(image_input):
    image_input=Input(shape=image_input)
    with tf.device('/cpu:0'):
        resnet_model = ResNet50(input_tensor=image_input,include_top=False, weights='imagenet')
    x = resnet_model.get_layer('res2a_branch2a').output
    #x = Conv2D(16, (3,3), padding="same", activation="relu")(x)
    resnet_model.trainable = False
    with tf.device('/cpu:0'):
        model=Model(inputs=resnet_model.input, outputs=x)
    return multi_gpu_model(model,gpus=2)

def build_imh(input_shape):

    def EM4(input,enhanced, channel):
        reshape1=Activation("relu")(Conv2D(32,(3,3),padding="same",data_format="channels_last")(Concatenate(axis=3)([enhanced,input])))
        #reshape=Activation("relu")(Conv2D(channel,(3,3),padding="same",data_format="channels_last")(input))
        reshape=keras.layers.MaxPool2D(pool_size=(3,3),strides=1,padding="same")(reshape1)
        #reshape2=Conv2D(channel,(3,3),activation="relu", padding="same",data_format="channels_last")(reshape)
        conv_1=Conv2D(32,kernel_size=(3,3),strides=1, padding="same")(reshape)
        conv_2=Activation("relu")(conv_1)
        conv_2=Conv2D(filters=32,kernel_size=(3,3),strides=1, padding="same")(conv_2)
        add_1=keras.layers.Add()([reshape, conv_2])#
        res1=Activation("relu")(add_1)

        #add_2=Concatenate(axis=3)([res1, input])
        #add_1 = Activation('relu')(BatchNormalization(axis=3)(add_1))
        #max_1=keras.layers.MaxPool2D(pool_size=(3,3),strides=1,padding="same")(res1)

        conv_3=Conv2D(filters=32,kernel_size=(3,3),strides=1, padding="same")(res1)
        conv_4=Activation("relu")(conv_3)
        conv_4=Conv2D(filters=32,kernel_size=(3,3),strides=1, padding="same")(conv_4)
        add_3=keras.layers.Add()([res1, conv_4])#
        res2=Activation("relu")(add_3)

        #add_4=Concatenate(axis=3)([res2,res1,enhanced, input])

        #dense_1=denseblock(add_4)

        #max_4=keras.layers.MaxPool2D(pool_size=(3,3),strides=1,padding="same")(add_4)

        #max_4=Conv2D(32,(3,3),padding="same",data_format="channels_last")(max_4)


        #add_2 = Activation('relu')(BatchNormalization(axis=3)(add_2))
        conv_5=Conv2D(filters=64,kernel_size=(3,3),strides=1, padding="same")(res2)
        conv_6=Activation("relu")(conv_5)
        conv_6=Conv2D(filters=64,kernel_size=(3,3),strides=1, padding="same")(conv_6)
        add_5=Concatenate(axis=3)([res2, res1, conv_6])#
        res3=Activation("relu")(add_5)




        conv_7=Conv2D(filters=64,kernel_size=(3,3),strides=1, padding="same")(res3)


        conv_10=Conv2D(3,(3,3),padding="same",data_format="channels_last")(conv_7)
        #res=keras.layers.Add()([conv_10, input])
        return Model(inputs=input,outputs=conv_10),conv_10


    def EM5(input,channel):
        reshape1=Activation("relu")(Conv2D(32,(3,3),padding="same",data_format="channels_last")(input))
        reshape=keras.layers.MaxPool2D(pool_size=(3,3),strides=1,padding="same")(reshape1)
        #reshape2=Conv2D(channel,(3,3),activation="relu", padding="same",data_format="channels_last")(reshape)
        conv_1=Conv2D(32,kernel_size=(3,3),strides=1, padding="same")(reshape)
        conv_2=Activation("relu")(conv_1)
        conv_2=Conv2D(filters=32,kernel_size=(3,3),strides=1, padding="same")(conv_2)
        add_1=keras.layers.Add()([reshape, conv_2])#
        res1=Activation("relu")(add_1)

        #add_2=Concatenate(axis=3)([res1, input])
        #add_1 = Activation('relu')(BatchNormalization(axis=3)(add_1))
        #max_1=keras.layers.MaxPool2D(pool_size=(3,3),strides=1,padding="same")(res1)

        conv_3=Conv2D(filters=32,kernel_size=(3,3),strides=1, padding="same")(res1)
        conv_4=Activation("relu")(conv_3)
        conv_4=Conv2D(filters=32,kernel_size=(3,3),strides=1, padding="same")(conv_4)
        add_3=keras.layers.Add()([res1, conv_4])#
        res2=Activation("relu")(add_3)

        #add_4=Concatenate(axis=3)([res2,res1,input])

        #dense_1=denseblock(add_4)


        #max_4=keras.layers.MaxPool2D(pool_size=(3,3),strides=1,padding="same")(add_4)

        #max_4=Conv2D(32,(3,3),padding="same",data_format="channels_last")(max_4)


        #add_2 = Activation('relu')(BatchNormalization(axis=3)(add_2))
        conv_5=Conv2D(filters=64,kernel_size=(3,3),strides=1, padding="same")(res2)
        conv_6=Activation("relu")(conv_5)
        conv_6=Conv2D(filters=64,kernel_size=(3,3),strides=1, padding="same")(conv_6)
        add_5=Concatenate(axis=3)([res2, res1, conv_6])#
        res3=Activation("relu")(add_5)


        #add_6=Concatenate(axis=3)([res3,reshape])
        conv_7=Conv2D(filters=64,kernel_size=(3,3),strides=1, padding="same")(res3)
        #conv_8=Conv2D(filters=128,kernel_size=(3,3),strides=1, padding="same")(conv_7)
        #conv_8=Conv2D(filters=16,kernel_size=(3,3),strides=1, padding="same")(conv_7)


        #add_3 = Activation('relu')(BatchNormalization(axis=3)(add_3))

        conv_10=Conv2D(3,(3,3),padding="same",data_format="channels_last")(conv_7)
        #add_6=Concatenate(axis=3)([res3,add_4])
        #max_5=keras.layers.MaxPool2D(pool_size=(3,3),strides=1,padding="same")(add_5)

        #add_3 = Activation('relu')(BatchNormalization(axis=3)(add_3))
        #conv_10=Conv2D(3,(3,3),padding="same",data_format="channels_last")(res3)
        #res=keras.layers.Add()([conv_10, input])
        return Model(inputs=input,outputs=conv_10),conv_10#


    inputs=Input(shape=(256,256,3))
    model_1,res_1 = EM5(inputs,16)
    model_2,res_2 = EM4(inputs, res_1, 16)
    model_3,res_3 = EM4(inputs, res_2, 16)
    with tf.device('/cpu:0'):
        model=Model(inputs,outputs=[res_1, res_2,res_3])
    return multi_gpu_model(model,gpus=2)
