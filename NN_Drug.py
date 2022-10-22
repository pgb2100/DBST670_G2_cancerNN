import sys

# data handling
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import MinMaxScaler

# feature selection
from sklearn.feature_selection import mutual_info_classif

# classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier

# neural building Libraries
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense


# performance metrics
from sklearn.metrics import balanced_accuracy_score,f1_score,precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import roc_auc_score

def main():
    # storing the data
    dataframe=pd.read_csv(file)

    # let's check the number of samples and features
    # note:the last column contain the labels. it is not considered as a feature

    print(dataframe.shape)
    g=[i for i in datanul if i>0]

    print('columns with missing values:%d'%len(g))

    # Checking for any null values in our dataset
    dataframe.isnull().sum()

    # Dropping The rows that have the null values so our data may be accurate as possible
    dataframe = dataframe.dropna(axis=0, how = "any", inplace=False)
    dataframe

    # Checking if the Null Values have been dropped
    dataframe.isnull().any()


    dataframe["Drug Name"].value_counts()

    # In our data we will use the following columns to develop our model that will be assigned to x and y variables


    X=dataframe[["Drug ID"]].values
    y= dataframe[["IC50"]].values

    #Encode labels
    #The labels for this data are categorical and we therefore have to convert them to numeric forms. 
    #This is referred to as encoding. Machine learning models usually require input data to be in numeric forms.
    label_encoder=LabelEncoder()
    label_encoder.fit(X)
    y=label_encoder.transform(X)
    labels=label_encoder.classes_
    classes=np.unique(X)
    nclasses=np.unique(X).shape[0]
    nclasses

    # split data into training,validation and test sets

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.8,random_state=1)

    # split the training set into two (training and validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.8)

    #Data Normalization
    #Data normalization is done so the values are in the same range to improve model performance and avoid bias.

    min_max_scaler=MinMaxScaler()
    X_train=min_max_scaler.fit_transform(X_train)
    X_val=min_max_scaler.fit_transform(X_val)
    X_test=min_max_scaler.fit_transform(X_test)
    
    # define model
    model = Sequential()

    # hidden layer 1
    model.add(Dense(40, input_dim=X_train.shape[1], activation='relu'))

    # hidden layer 2
    model.add(Dense(20, activation='relu'))

    # output layer
    model.add(Dense(nclasses, activation='softmax'))

    # define optimizer and learning rate. We will use Adam optimizer
    opt_adam = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt_adam, metrics=[keras.metrics.SparseCategoricalAccuracy()])

    # fit the model to the training data
    # the higher the epoch value the higher the accuracy but the longer it takes to execute. 
    # An epoch means training the neural network with all the training data for one cycle. 
    # In an epoch, we use all of the data exactly once. A forward pass and a backward pass together are counted as one pass: 
    # An epoch is made up of one or more batches, where we use a part of the dataset to train the neural network.
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=30,epochs=32, verbose=1)

    predictions = model.predict(X_test)
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)

    accuracy

    predictions[0]

    np.argmax(predictions)

    # Get the predictions for samples in the test set. You can change by alter the number within the []
    for index,entry in enumerate(predictions[0:30,:]):
        print('predicted:%d ,actual:%d'%(np.argmax(entry),y_test[index]))

    # Summarize history for accuracy
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('model performance')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.show()

    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.show()

if __name__ == '__main__':
    # read data directly from my home computer or GDRIVE
    file=sys.argv[1]
    main()