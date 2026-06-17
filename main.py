import tensorflow as tf
from nn import network
import pandas as pd
import numpy as np
(trainx, trainy),(testx,testy)=tf.keras.datasets.mnist.load_data()
trainX=trainx.reshape(-1,784)/255.0
testX=testx.reshape(-1,784)/255.0

net1=network()
# _,_,_,_,_,_,y=net1.forward_p(trainX)




#
cats=np.unique(trainy)
ohe=(trainy[:,None]==cats).astype(int)
print(trainx.shape, trainy.shape,ohe.shape)
trainx=trainx.reshape(-1,784)
net1.train(1000,trainx,ohe,0.001)

