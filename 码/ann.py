import scipy.io as sio
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential

from keras import optimizers
import keras
matfn=u'C:/Users/hang/Desktop/file/ee559/project/feature.mat'

data=sio.loadmat(matfn)
hangtr=data['feature_train']
hangte=data['feature_test']
hangtr_label=data['label_train']
hangte_label=data['label_test']

hangtr_label=hangtr_label-1
hangte_label=hangte_label-1

classnum=5
hang_iter=1
hang_block=32
hang_wd=0.0001
hang_rush=0.90
hang_rate=0.001
hangtr_label = keras.utils.to_categorical(hangtr_label, classnum)
hangte_label = keras.utils.to_categorical(hangte_label, classnum)

acc_trace=[]
let1=[]
let2=[]
for i in range(10,50,10):
    for j in range(10,50,10):
        def Lenet5Model():
            lenet5Model = Sequential()
            input_shape =hangtr[0].shape
            lenet5Model.add(Dense(i, activation='relu',input_shape=input_shape))
            lenet5Model.add(Dense(j, activation='relu'))
           
            lenet5Model.add(Dense(5, activation='softmax'))
            #lenet5Model.build((None,90, 7))
            lenet5Model.summary()
            return lenet5Model



        
        hang_ann = Lenet5Model()

        sgd = optimizers.SGD(lr=hang_rate, decay=hang_wd, momentum=hang_rush, nesterov=True)
        hang_ann.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])





        hang_stru = hang_ann.fit(hangtr, hangtr_label,
                                     validation_data=(hangte, hangte_label),
                                     batch_size=hang_block,
                                     epochs=hang_iter)
        
        
        
        
        hang_acc= hang_ann.evaluate(hangte, hangte_label)
        
        print('The accuracy of model is:', hang_acc[1])
        
        acc_trace.append(hang_acc)
    let1.append(i)
    let2.append(j)
        
"""
trainAcc = hang_stru.history['accuracy']
testAcc = hang_stru.history['val_accuracy']
# train time stop

a = np.arange(0, hang_iter)
plt.figure()
plt.title('train and test accuracy')
plt.plot(a, trainAcc)
plt.plot(a, testAcc)
"""