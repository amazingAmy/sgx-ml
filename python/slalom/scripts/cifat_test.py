import tensorflow as tf
from keras.datasets import cifar10
import keras
from keras.layers import InputLayer,Conv2D,MaxPool2D,Dense,Flatten
from keras.models import Sequential
from python.slalom.quant_layers import transform
from python.slalom.sgxdnn import SGXDNNUtils,model_to_json

def main():
    model = Sequential()
    model.add(InputLayer(input_shape=[32,32,3]))
    model.add(Conv2D(64,kernel_size=3,strides=1,padding='same',activation='relu'))
    model.add(Conv2D(64,kernel_size=3,strides=1,padding='same',activation='relu'))
    model.add(MaxPool2D(pool_size=2,strides=2,padding='same'))

    model.add(Conv2D(128,kernel_size=3,strides=1,padding='same',activation='relu'))
    model.add(Conv2D(128,kernel_size=3,strides=1,padding='same',activation='relu'))
    model.add(MaxPool2D(pool_size=2,strides=2,padding='same'))

    model.add(Conv2D(256,kernel_size=3,strides=1,padding='same',activation='relu'))
    model.add(Conv2D(256,kernel_size=3,strides=1,padding='same',activation='relu'))
    model.add(MaxPool2D(pool_size=2,strides=2,padding='same'))

    model.add(Flatten())

    model.add(Dense(4096,activation='relu'))
    model.add(Dense(10,activation='softmax'))

    sgd = keras.optimizers.SGD(lr=0.001,decay=0.0,nesterov=False)
    model.compile(sgd,'categorical_crossentropy',metrics=['accuracy'])
    model.summary()

