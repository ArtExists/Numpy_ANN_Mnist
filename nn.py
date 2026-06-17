import numpy as np
import random


class network:
    def __init__(self):
        self.weights=np.random.randn(784,128)*np.sqrt(2/784)
        self.hw=np.random.randn(128,64)*np.sqrt(2/128)
        self.hw2=np.random.randn(64,10)*np.sqrt(2/64)

        self.b1=np.zeros((1,128))
        self.b2=np.zeros((1,64))
        self.b3=np.zeros((1,10))
    def softie(self,x):
        exp=np.exp(x-np.max(x,axis=1,keepdims=True))
        return exp/np.sum(exp,axis=1,keepdims=True)

    def ReLU(self,x):
        return np.maximum(0,x)

    def forward_p(self,x):
        inp = x@self.weights+self.b1
        a1= self.ReLU(inp)
        hl= a1@self.hw+self.b2
        a2= self.ReLU(hl)
        hl2=a2@self.hw2+self.b3
        a3=self.softie(hl2)

        return x,inp,hl,hl2,a1,a2,a3
    # def ReLU_diff(self,x):
    #     y=np.zeros_like(x)
    #     (i,j)=x.shape
    #     for a in range(i):
    #         for b in range (j):
    #             if x[a,b]>0:
    #                 y[a,b]=1
    #
    #     return y
    def ReLU_diff(self, x):
        return (x > 0).astype(float)


    def cce_loss(self, y, yh):
        print("acc: ",np.mean(np.argmax(yh,axis=1)==np.argmax(y,axis=1)))
        return -np.mean(np.sum(y*np.log(yh+1e-9),axis=1))
    # def soft_diff(self, z, loss):
    #     top=
    #     # top=np.sum(np.exp(z))*(np.exp(z)) -



    def back_prop(self, data, y, lr):



        x,inp,hl1,hl2,a1,a2,yh=self.forward_p(data)
        loss=self.cce_loss(y, yh)
        print(loss)

        m=x.shape[0]

        #soft_diff_top= (y/yh)*(np.sum(np.exp(hl2))@np.exp(hl2) - np.exp(hl2)*np.sum(np.exp(hl2)*self.hw2))    #dloss/dhw2
        dy3=yh-y
        dy2 = (yh - y) @ self.hw2.T*self.ReLU_diff(hl1)
        dy1 = dy2 @ self.hw.T*self.ReLU_diff(inp) #direct multiply cuz its basically like a masking matrix over existing weights (like wherever you didnt get any values youll put 0 there kinda like bitmask
        db1=(np.sum(dy1,axis=0,keepdims=True))/m
        db2 = (np.sum(dy2, axis=0, keepdims=True))/m
        db3 = (np.sum(dy3, axis=0, keepdims=True))/m
        dw2 = (a1.T @ dy2)/m
        dw1 = (x.T @ dy1)/m
        dw3=(a2.T@dy3)/m
        self.hw2=self.hw2-lr*dw3
        self.b3=self.b3-lr*db3


        #next wt


        self.hw = self.hw - lr * dw2
        self.b2 = self.b2 - lr * db2


        #nextt




        self.weights = self.weights - lr * dw1
        self.b1 = self.b1 - lr * db1


    def train(self,epochs,x,ytrain,lr):
        for i in range(epochs):
            #xo,inp,hl,hl2,a1,a2,a3=self.forward_p(x)
            #print("Loss: ",self.cce_loss(ytrain,a3))
            self.back_prop(x,ytrain,lr)






if __name__=="__main__":
        net1=network()

        print(net1.weights)