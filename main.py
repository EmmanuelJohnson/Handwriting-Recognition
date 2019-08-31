#!/usr/bin/env python
# coding: utf-8
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os, glob, gzip, pickle

from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns; sns.set()

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, Flatten
from keras.models import load_model

#Configuration for logistic regression
#learning rate
EETA = 0.04
#regularizer
LAMDA = 0.001
L_EPOCHS = 3500

#Save the pickle data
def save_pickle_data(pickleData,filename):
    with open(filename, 'wb') as f:
        pickle.dump(pickleData, f)
    f.close()

#Get data from pickle file
def get_pikle_data(filename):
    pkl = None
    with open(filename, 'rb') as f:
        pkl = pickle.load(f)
    f.close()
    return pkl

#Get dataset of mnist
def unpack_mnist_data(filename):
    mnist = dict()
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    mnist['training'] = training_data
    mnist['validation'] = validation_data
    mnist['testing'] = test_data
    df = pd.DataFrame.from_dict(mnist)
    df.to_csv('mnist.csv')
    return mnist

#Get dataset of usps
def unpack_usps_data(directory):
    usps = dict()
    uspsMat  = []
    uspsTar  = []
    #resImg = []
    for ipath in tqdm(glob.glob(directory+"/*/*")):
        indexFolder = int(ipath.split('/')[2])
        fileType = ipath[-3:]
        if fileType == 'png':
            img = Image.open(ipath,'r')
            resizedImg = img.resize((28, 28))
            img.close()
            imgData = (255-np.array(resizedImg.getdata()))/255
            #resImg.append(resizedImg)
            uspsMat.append(imgData)
            uspsTar.append(indexFolder)
    usps['imgMatrix'] = uspsMat
    usps['indexFolders'] = uspsTar
    #usps['resizedImage'] = resImg
    return usps

#Plot a heatmap of the confusion matrix
def plot_confusion_matrix(conf_matrix, filename):
    sns.heatmap(conf_matrix.T, square=True, annot=True, fmt='d', cbar=False, cmap="YlGnBu")
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.savefig(filename+'.png')
    plt.clf()

#Softmax expression
def softmax(z):
    exps = np.exp(z)
    sum_exps = exps.sum(axis=1)
    return (exps.T/sum_exps).T

def amax(z):
    m = np.argmax(z,axis=1)
    return np_utils.to_categorical(m,10)

#Logistic regression loss
def logisticLoss(h,y):
    logh = np.log(h)
    hy = logh*y
    return (-1/h.shape[0])*(np.sum(hy))

def getLogisticIterAccuracy(ycap,y):
    predictions = amax(ycap)
    correct = 0
    for yc,yo in zip(predictions,y):
        if np.argmax(yc) == np.argmax(yo):
            correct+=1
    accuracy = (correct/len(ycap)) * 100
    return accuracy

def getLogisticAccuracy(x,y,w):
    h = softmax(np.dot(x,w))
    predictions = amax(h)
    correct = 0
    for yc,yo in zip(predictions,y):
        if np.argmax(yc) == np.argmax(yo):
            correct+=1
    accuracy = (correct/len(y)) * 100
    cm = confusion_matrix(y.argmax(axis=1), predictions.argmax(axis=1))
    return accuracy,cm

def getAccuracy(ycap,y):
    correct = 0
    for yc,yo in zip(ycap,y):
        if np.argmax(yc) == np.argmax(yo):
            correct+=1
    accuracy = (correct/len(y)) * 100
    cm = confusion_matrix(y, ycap)
    return accuracy,cm

def logisticRegression(mdata):
    case = str(EETA)+"_"+str(L_EPOCHS)+"_"
    iterVals = {'training':{'acc':list(),'loss':list()},'validation':{'acc':list(),'loss':list()},'testing':{'acc':list(),'loss':list()}}

    trainFeatures = mdata['training']['input']
    print('Traing Features', trainFeatures.shape)
    weights = np.zeros((trainFeatures.shape[1],10))
    print('Weights', weights.shape)
    for step in tqdm(range(L_EPOCHS)):
        z = np.dot(trainFeatures, weights)
        
        predictions = softmax(z)
        trainTarget = mdata['training']['target']
        
        errorDiff =  predictions - trainTarget
        #Gradient descent
        gradient = (np.dot(trainFeatures.T, errorDiff)/trainTarget.size) + (LAMDA*weights)
        #Update the weights
        weights = weights - (EETA * gradient)

        trainLoss = logisticLoss(predictions,trainTarget)
        trainAcc = getLogisticIterAccuracy(predictions,trainTarget)
        iterVals['training']['acc'].append(trainAcc)
        iterVals['training']['loss'].append(trainLoss)
        
        z = np.dot(mdata['validation']['input'], weights)
        predictions = softmax(z)
        valLoss = logisticLoss(predictions,mdata['validation']['target'])
        valAcc = getLogisticIterAccuracy(predictions,mdata['validation']['target'])
        iterVals['validation']['acc'].append(valAcc)
        iterVals['validation']['loss'].append(valLoss)
        
        z = np.dot(mdata['testing']['input'], weights)
        predictions = softmax(z)
        testLoss = logisticLoss(predictions,mdata['testing']['target'])
        testAcc = getLogisticIterAccuracy(predictions,mdata['testing']['target'])
        iterVals['testing']['acc'].append(testAcc)
        iterVals['testing']['loss'].append(testLoss)

    print(getLogisticAccuracy(mdata['testing']['input'],mdata['testing']['target'],weights))
    
    df = {}
    df['Training Loss'] = iterVals['training']['loss']
    plt.plot(iterVals['training']['loss'])
    
    df['Validation Loss'] = iterVals['validation']['loss']
    df['Testing Loss'] = iterVals['testing']['loss']
    
    plt.savefig('loss_'+case+'.png')
    plt.clf()

    df['Training Acc'] = iterVals['training']['acc']
    plt.plot(iterVals['training']['acc'])

    df['Validation Acc'] = iterVals['validation']['acc']
    df['Testing Acc'] = iterVals['testing']['acc']

    plt.savefig('accuracy_'+case+'.png')
    plt.clf()

    pd.DataFrame(df).to_csv('logistic_'+case+'.csv')
    print('Training Accuracy: ', np.around(max(iterVals['training']['acc']),5), 'and Training Loss : ', np.around(min(iterVals['training']['loss']),5))
    print('Validation Accuracy : ', np.around(max(iterVals['validation']['acc']),5), 'and Validation Loss : ', np.around(min(iterVals['validation']['loss']),5))
    print('Testing Accuracy : ', np.around(max(iterVals['testing']['acc']),5), 'and Testing Loss : ', np.around(min(iterVals['testing']['loss']),5))
    return weights

def startLogisticModel(mdata,udata):
    #Check if pickle file is availabe
    if not os.path.exists('optimal/logistic_weights.pkl'):
        lweights = logisticRegression(mdata)
        save_pickle_data(lweights, 'optimal/logistic_weights.pkl')
    else:
        lweights = get_pikle_data('optimal/logistic_weights.pkl')

    accuracy, conf_matrix = getLogisticAccuracy(mdata['testing']['input'],mdata['testing']['target'],lweights)
    print('MNIST Accuracy', accuracy)
    print('MNIST Confusion Matrix', conf_matrix)
    plot_confusion_matrix(conf_matrix,'logistic_mnist_confusion_matrix')
    accuracy, conf_matrix = getLogisticAccuracy(udata['testing']['input'],udata['testing']['target'],lweights)
    print('USPS Accuracy', accuracy)
    print('USPS Confusion Matrix', conf_matrix)
    plot_confusion_matrix(conf_matrix,'logistic_usps_confusion_matrix')
    

def startRandomForest(mdata,udata):
    #Check if pickle file is available
    if not os.path.exists('optimal/rf_weights.pkl'):
        rf = RandomForestClassifier(n_estimators=100, verbose=2, n_jobs=-1)
        rf.fit(mdata['training']['input'], mdata['training']['target'])
    else:
        rf = get_pikle_data('optimal/rf_weights.pkl')

    predictions = rf.predict(mdata['testing']['input'])
    conf_matrix = confusion_matrix(mdata['testing']['target'], predictions)
    plot_confusion_matrix(conf_matrix, 'rf_mnist_confusion_matrix')
    print('MNIST Accuracy', rf.score(mdata['testing']['input'],mdata['testing']['target']))
    print('MNIST Confustion Matrix', conf_matrix)
    predictions = rf.predict(udata['testing']['input'])
    conf_matrix = confusion_matrix(udata['testing']['target'], predictions)
    plot_confusion_matrix(conf_matrix, 'rf_usps__confusion_matrix')
    print('USPS Accuracy', rf.score(udata['testing']['input'],udata['testing']['target']))
    print('USPS Confustion Matrix', conf_matrix)

def startSVM(mdata,udata):
    #Check if pickle file is available
    if not os.path.exists('optimal/svm_weights.pkl'):
        svc = SVC(kernel='rbf', gamma=0.05, C=10, verbose=1)
        svc.fit(mdata['training']['input'], mdata['training']['target'])
        save_pickle_data(svc, 'optimal/svm_weights.pkl')
    else:
        svc = get_pikle_data('optimal/svm_weights.pkl')
    
    predictions = svc.predict(mdata['testing']['input'])
    conf_matrix = confusion_matrix(mdata['testing']['target'], predictions)
    plot_confusion_matrix(conf_matrix, 'svc_mnist_confusion_matrix')
    print('MNIST Accuracy',svc.score(mdata['testing']['input'],mdata['testing']['target']))
    print('MNIST Confustion Matrix', conf_matrix)
    predictions = svc.predict(udata['testing']['input'])
    conf_matrix = confusion_matrix(udata['testing']['target'], predictions)
    plot_confusion_matrix(conf_matrix, 'svc_usps_confusion_matrix')
    print('USPS Accuracy', svc.score(udata['testing']['input'],udata['testing']['target']))
    print('USPS Confustion Matrix', conf_matrix)
    
def getModel(inputData,targetData):
    input_size = inputData.shape[1]
    first_dense_layer_nodes  = 512
    second_dense_layer_nodes = 10
    model = Sequential()
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def startNeuralNetworks(mdata,udata):
    inputData = mdata['training']['input']
    targetData = mdata['training']['target']
    model = getModel(inputData, targetData)
    v_split = 0.1
    n_epochs = 125
    m_batchSize = 128
    history = model.fit(inputData, targetData, validation_split=v_split, epochs=n_epochs, batch_size=m_batchSize, verbose=1)
    model.save('optimal/nn_model.h5')
    df = pd.DataFrame(history.history)
    df.plot(subplots=True, grid=True, figsize=(5,10))
    plt.savefig('nn_graph.png')

    predictions = model.predict(mdata['testing']['input'])
    conf_matrix = confusion_matrix(mdata['testing']['target'].argmax(axis=1), predictions.argmax(axis=1))
    plot_confusion_matrix(conf_matrix, 'nn_mnist_confusion_matrix')
    loss,accuracy = model.evaluate(mdata['testing']['input'], mdata['testing']['target'], verbose=False) 
    print('MNIST Accuracy', accuracy)
    print('MNIST Confustion Matrix', conf_matrix)

    predictions = model.predict(udata['testing']['input'])
    conf_matrix = confusion_matrix(udata['testing']['target'].argmax(axis=1), predictions.argmax(axis=1))
    plot_confusion_matrix(conf_matrix, 'nn_usps_confusion_matrix')
    loss,accuracy = model.evaluate(udata['testing']['input'], udata['testing']['target'], verbose=False) 
    print('USPS Accuracy', accuracy)
    print('USPS Confustion Matrix', conf_matrix)

#https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5
def startConvolutionNN(mdata,udata):
    #Reshape the data
    inputData = mdata['training']['input'].reshape(mdata['training']['input'].shape[0],28,28,1)
    testData = mdata['testing']['input'].reshape(mdata['testing']['input'].shape[0],28,28,1)
    utestData = udata['testing']['input'].reshape(udata['testing']['input'].shape[0],28,28,1)
    
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(inputData, mdata['training']['target'], validation_split=0.1, epochs=3, verbose=1)
    print('MNIST')
    loss,accuracy = model.evaluate(testData, mdata['testing']['target'], verbose=False) 
    print('Loss: ',loss,' Accuracy: ',accuracy)
    print('USPS')
    loss,accuracy = model.evaluate(utestData, udata['testing']['target'], verbose=False) 
    print('Loss: ',loss,' Accuracy: ',accuracy)

def preprocessMNIST(data,splits,cat):
    allData = np.concatenate((data['training'][0], data['validation'][0],data['testing'][0]), axis=0)
    allTarg = np.concatenate((data['training'][1], data['validation'][1],data['testing'][1]), axis=0)
    if cat:
        allTarg = np_utils.to_categorical(allTarg,10)
    
    split = np.split(allData, splits)
    split_t = np.split(allTarg, splits)

    modelingData = dict()
    if len(splits) == 3:
        tr, vl, ts = split[0], split[1], split[2]
        tr_t, vl_t, ts_t = split_t[0], split_t[1], split_t[2]
        modelingData = {'training':{ 'input': tr, 'target': tr_t},
                    'validation':{ 'input': vl, 'target': vl_t},
                    'testing':{ 'input': ts, 'target': ts_t}}
    else:
        tr, ts = split[0], split[1]
        tr_t, ts_t = split_t[0], split_t[1]
        modelingData = {'training':{ 'input': tr, 'target': tr_t},
                    'testing':{ 'input': ts, 'target': ts_t}}    
    return modelingData

def preprocessUSPS(data,cat):
    inputData = np.array(data['imgMatrix'])
    targetData = np.array(data['indexFolders'])
    if cat:
        targetData = np_utils.to_categorical(targetData,10)
    modelingData = {'testing':{ 'input': inputData, 'target': targetData}}
    return modelingData    

def ensemble(ldata,mdata,ndata,t):
    print('Logistic predictions')
    logistic_model = get_pikle_data('optimal/logistic_weights.pkl')
    h = softmax(np.dot(ldata['testing']['input'],logistic_model))
    l_predictions = amax(h)
    l_y = ldata['testing']['target']
    l_cm = confusion_matrix(l_y.argmax(axis=1), l_predictions.argmax(axis=1))
    l_predictions = l_predictions.argmax(axis=1)
    
    print('random forest prediction')
    rf_model = get_pikle_data('optimal/rf_weights.pkl')
    r_predictions = rf_model.predict(mdata['testing']['input'])
    r_cm = confusion_matrix(mdata['testing']['target'], r_predictions)

    print('svm predictions')
    s_predictions = get_pikle_data('optimal/svm_pred_'+t+'.pkl')
    s_cm = confusion_matrix(mdata['testing']['target'], s_predictions)

    print('neural predictions')
    nn_model = load_model('optimal/nn_model.h5')
    n_predictions = nn_model.predict(ndata['testing']['input'])
    n_cm = confusion_matrix(ndata['testing']['target'].argmax(axis=1), n_predictions.argmax(axis=1))
    n_predictions = n_predictions.argmax(axis=1)

    epredictions = list()

    for lp, rp, sp, nnp in zip(l_predictions,r_predictions,s_predictions,n_predictions):
        row = [lp,rp,sp,nnp]
        #Get the mode of the list which the maximum number of occurences
        highest_voting = max(set(row), key=row.count)
        epredictions.append(highest_voting)
    epredictions = np.array(epredictions)
    e_predictions = np_utils.to_categorical(epredictions,10)
    e_cm = confusion_matrix(mdata['testing']['target'], e_predictions.argmax(axis=1))
    e_accuracy = (e_cm.diagonal().sum())/e_cm.sum()
    plot_confusion_matrix(e_cm, t+'_confusion_matrix')
    print('Accuracy after majority voting: ', e_accuracy)
    print('Confusion matrix after majority voting', e_cm)


if __name__ == '__main__':
    
    #Unpack MNIST Dataset
    mnist = unpack_mnist_data('mnist.pkl.gz')

    usps_pkl_path = 'usps.pkl'
    if not os.path.exists(usps_pkl_path):
        #Unpack USPS Dataset
        usps = unpack_usps_data('USPSdata/Numerals')
        save_pickle_data(usps, usps_pkl_path)
    else:
        usps = get_pikle_data(usps_pkl_path)

    #If true make the target values as one hot vector
    log_mdata = preprocessMNIST(mnist,[50000, 60000, 70000], True)
    mdata = preprocessMNIST(mnist,[60000, 70000], False)
    neural_mdata = preprocessMNIST(mnist,[60000, 70000], True)

    c_udata = preprocessUSPS(usps, True)
    udata = preprocessUSPS(usps, False)
    usps.clear()
    mnist.clear()

    print('__Logistic Regression__')
    startLogisticModel(log_mdata,c_udata)

    print('__Random Forest__')
    startRandomForest(mdata,udata)

    print('__Support Vector Machine__')
    startSVM(mdata,udata)

    print('__Deep Neural Network__')
    startNeuralNetworks(neural_mdata,c_udata)

    print('__Convolution Neural Network__')
    startConvolutionNN(neural_mdata,c_udata)

    print('ENSEMBLE MNIST')
    ensemble(log_mdata,mdata,neural_mdata,'m')

    print('ENSEMBLE USPS')
    ensemble(c_udata,udata,c_udata,'u')

    
