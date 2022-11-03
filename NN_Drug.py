import sys

# data handling libraries
import pandas as pd
import numpy as np

# preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# neural building Libraries
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

# data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # storing the data
    df=pd.read_csv(file)

    ## Data Exploration & Cleaning
    #Data cleaning is very important in machine learning as it makes our data to be of a high quality for better results.
    #Here we will check for Null values and remove them for greater accuracy\
    #In the exploration stage we will see the structure of our data, the columns, rows and their numbers
    df.head(50)
    # Checking for any null values in our dataset
    df.isnull().sum()
    
    #Data Exploration & Cleaning
    #Data cleaning is very important in machine learning as it makes our data to be of a high quality for better results.
    #Dropping the null values
    # Dropping The rows that have the null values so our data may be accurate as possible
    df = df.dropna(axis=0, how = "any", inplace=False)
    df

    # Checking if the Null Values have been dropped
    df.isnull().any()

    ## Checking for the drugs that our data set contains
    df["gene_1000"].value_counts()

    #Checking for the drugs that our data set contains
    dfa["IC50"].value_counts()
    dfb["gene_1000"].value_counts()
    
    #Data preprocesing
    #This is done to put the data in an appropriate format before modelling
    # in our data we will used= the following columns to develop our model that will be assigned to x and y variables
    
    Y=df[["IC50"]].values
    X= df[["gene_1000"]].values

    #Encode labels & Data Splitting
    #The labels for this data are categorical and we therefore have to convert them to numeric forms. This is referred to as encoding. Machine learning models usually require input data to be in numeric forms.
    #Data is split into three: training, validation and test sets
    #validation set is used for evaluating the model during training.
    #training set is used for training
    #test set is used to test the model after training and tuning has been completed.

    # let's encode target labels (y) with values between 0 and n_classes-1.
    # we will be using LabelEncoder to perform the encoding

    label_encoder=LabelEncoder()
    label_encoder.fit(X)
    y=label_encoder.transform(X)
    labels=label_encoder.classes_
    classes=np.unique(X)
    nclasses=np.unique(X).shape[0]
    nclasses

    #split data into training,validation and test sets
    X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3,random_state=1)
    
    #split the training set into two (training and validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.3)
    
    # check the shape of X_train and y_train
    X_train.shape, y_train.shape

    #Data Normalization
    #Data normalization is done so the values are in the same range to improve model performance and avoid bias.

    min_max_scaler=MinMaxScaler()
    X_train=min_max_scaler.fit_transform(X_train)
    X_val=min_max_scaler.fit_transform(X_val)
    X_test=min_max_scaler.fit_transform(X_test)

    #Building Neural Network
    # define model
    model = keras.Sequential()

    # hidden layer 1
    model.add(Dense(60, input_dim=X_train.shape[1], activation='relu'))

    # hidden layer 2
    model.add(Dense(60, activation='relu'))

    # hidden layer 3
    model.add(Dense(40, activation='relu'))

    # output layer
    model.add(Dense(nclasses, activation='softmax'))

    # define optimizer and learning rate. We will use Adam optimizer
    opt_adam = keras.optimizers.Adam(learning_rate=0.003)

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt_adam, metrics=[keras.metrics.SparseCategoricalAccuracy()])

    # fit the model to the training data
    # the higher the epoch value the higher the accuracy but the longer it takes to execute. 
    # An epoch means training the neural network with all the training data for one cycle. 
    # In an epoch, we use all of the data exactly once. A forward pass and a backward pass together are counted as one pass: 
    # An epoch is made up of one or more batches, where we use a part of the dataset to train the neural network.

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32,epochs=64, verbose=1)

    print(X_test)

    print(y_test)

    Classification_Model=Sequential()

    prediction= Classification_Model.predict(X_test)
    prediction=[1 if x>0.5 else 0 for x in prediction]#list
    print(prediction)

    predictions = model.predict(X_test)
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(accuracy)

    print(predictions[0])

    np.argmax(predictions)

    # Get the predictions for samples in the test set. You can change by alter the number within the []
    for index,entry in enumerate(predictions[0:10,:]):
        print('predicted:%d ,actual:%d'%(np.argmax(entry),y_test[index]))

    
    # Summarize history for accuracy
    plt.clf()
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('model performance')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.savefig("history_for_accuracy.png")


    # Summarize history for loss
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.savefig("history_for_loss.png")

    # plot histogram for the data
    plt.clf()
    df['gene_1000'].plot(kind='hist')
    df['IC50'].plot(kind='hist')
    plt.savefig("data_histogram.png")


if __name__ == '__main__':
    file = sys.argv[1]
    main()