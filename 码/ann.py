import scipy.io as sio
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from keras.layers import Dropout
from keras import optimizers
import keras
import seaborn as sns

matfn=u'C:/Users/hang/Desktop/file/ee559/project/feature.mat'

data=sio.loadmat(matfn)
hangtr=data['feature_train']
hangte=data['feature_test']
hangtr_label_all=data['label_train']
hangte_label_all=data['label_test']

hangtr_label_all=hangtr_label_all-1
hangte_label_all=hangte_label_all-1


acc_trace=[]
let1=[]
let2=[]

dr=0.2
def Lenet5Model():
    lenet5Model = Sequential()
    input_shape =hangtr[0].shape
    lenet5Model.add(Dense(500, activation='relu',input_shape=input_shape))
    lenet5Model.add(Dropout(p = dr)) # Disable 10% of the neurons on each iteration
    lenet5Model.add(Dense(400, activation='relu'))
    lenet5Model.add(Dropout(p = dr)) # Disable 10% of the neurons on each iteration
    lenet5Model.add(Dense(300, activation='relu'))   
    lenet5Model.add(Dropout(p = dr)) # Disable 10% of the neurons on each iteration
    lenet5Model.add(Dense(200, activation='relu'))   
    lenet5Model.add(Dropout(p = dr)) # Disable 10% of the neurons on each iteration
    lenet5Model.add(Dense(100, activation='relu'))   
    lenet5Model.add(Dropout(p = dr)) # Disable 10% of the neurons on each iteration
    lenet5Model.add(Dense(50, activation='relu'))   
    lenet5Model.add(Dropout(p = dr)) # Disable 10% of the neurons on each iteration
    lenet5Model.add(Dense(5, activation='softmax'))
    #lenet5Model.build((None,90, 7))
    lenet5Model.summary()
    return lenet5Model


classnum=5
hang_iter=200
hang_block=64
hang_wd=0.0001
hang_rush=0.90
hang_rate=0.00015
        
hang_ann = Lenet5Model()
hangtr_label = keras.utils.to_categorical(hangtr_label_all, classnum)
hangte_label = keras.utils.to_categorical(hangte_label_all, classnum)
sgd = optimizers.SGD(lr=hang_rate, decay=hang_wd, momentum=hang_rush, nesterov=True)
hang_ann.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])





hang_stru = hang_ann.fit(hangtr, hangtr_label,
                             validation_data=(hangte, hangte_label),
                             batch_size=hang_block,
                             epochs=hang_iter)




hang_acc= hang_ann.evaluate(hangte, hangte_label)

print('The accuracy of model is:', hang_acc[1])

acc_trace.append(hang_acc)
trainAcc = hang_stru.history['accuracy']
testAcc = hang_stru.history['val_accuracy']
# train time stop

a = np.arange(0, hang_iter)
plt.figure()
plt.title('train and test accuracy')
plt.plot(a, trainAcc)
plt.plot(a, testAcc)


predict_label = hang_ann.predict_classes(hangte)
hang_cfmatrix = confusion_matrix(hangte_label_all, predict_label,normalize='true')
print("confusion maxtrix:", hang_cfmatrix)
hang_name = (['posture 1','posture 2','posture 3','posture 4','posture 5',])
fig = plt.figure(figsize=[10, 6])
sns_plot = sns.heatmap(hang_cfmatrix, annot=True, xticklabels=hang_name, yticklabels=hang_name)
plt.title('Confusion Matrix')
