# Importing libraries
import sys
import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

def main():
    #loed dataset & choose the column
    data=pd.read_csv(file)
    df=data[['Drug Name','Cell Line Name','Tissue','Tissue Sub-type','IC50']]
    df

    #label encoder
    encoder = LabelEncoder()
    dframe=pd.DataFrame()
    dframe["Drug Name"] = encoder.fit_transform(df["Drug Name"])
    dframe["Cell Line Name"] = encoder.fit_transform(df["Cell Line Name"])
    dframe["Tissue"] = encoder.fit_transform(df["Tissue"])
    dframe["Tissue Sub-type"] = encoder.fit_transform(df["Tissue Sub-type"])
    dframe['IC50']=df['IC50']
    dframe

    # get the correlation matrix
    dframe.corr()

    #plot histogram for the data
    plt.figure(figsize=(10,10))
    dframe['Drug Name'].plot(kind='hist',color="green")
    dframe['Tissue'].plot(kind='hist',color="yellow")
    dframe['Tissue Sub-type'].plot(kind='hist',color="orange")
    dframe['IC50'].plot(kind='hist',color='purple')
    dframe['Cell Line Name'].plot
    plt.savefig("data_histogram.png")

    # plot a scatter plot
    plt.clf()
    plt.scatter(x=dframe['Tissue Sub-type'],y=dframe['Drug Name'])
    plt.scatter(x=dframe['Tissue'],y=dframe['Drug Name'])
    plt.scatter(x=dframe['Cell Line Name'],y=dframe['Drug Name'])
    plt.scatter(x=dframe['Drug Name'],y=dframe['Drug Name'])
    plt.scatter(x=dframe['Tissue Sub-type'],y=dframe['Drug Name'])
    plt.savefig("data_scatter.png")

    #plot a heatmap
    dataframe= np.random.rand( 10 , 10 )
    ax = sns.heatmap( dataframe , linewidth = 1.5 , cmap = 'coolwarm' )
    plt.clf()
    plt.title( "Random Forest For Drug Efficiency Against Tumors" )
    plt.savefig("data_heatmap.png")

    #X and Y
    X=dframe[['Cell Line Name','Tissue','Tissue Sub-type','IC50']]
    Y=dframe[['Drug Name']]

    #scaling
    scaler=StandardScaler()
    X=scaler.fit_transform(X)
    print(X)

    #Split the dataset
    X_train, X_test, y_train, y_test =train_test_split(X, Y, test_size = 0.1, random_state = 24)
    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Test: {X_test.shape}, {y_test.shape}")

    # Training and testing Random Forest Classifier
    rf_model = RandomForestClassifier(random_state=18)
    rf_model.fit(X, Y)

    preds = rf_model.predict(X_test)
    print(preds)

    #predict a specific value
    scaled_x=scaler.transform([[203,4,50,10.541691]])
    print(scaled_x)

    predicted=rf_model.predict([scaled_x[0]])
    print(predicted)

    print(f"Accuracy on train data by Random Forest Classifier\: {accuracy_score(y_train, rf_model.predict(X_train))*100}")

    #Save the classifier
    joblib.dump(rf_model,'/Users/Albert3/Downloads/M_luad.joblib',compress=5)
    model=joblib.load('/Users/Albert3/Downloads/M_luad.joblib')
    tframe=df.join(dframe,rsuffix='enc')
    print(tframe)

    #predict actual values
    cln='Calu-6'
    tis='lung'
    subt='lung_NSCLC_adenocarcinoma'
    i50=3.009075
    test=pd.DataFrame()
    test['clnenc']=tframe['Cell Line Nameenc'].where((tframe['Cell Line Name']==cln))
    test['tis']=tframe['Tissueenc'].where((tframe['Tissue']==tis))
    test['subt']=tframe['Tissue Sub-typeenc'].where((tframe['Tissue Sub-type']==subt))
    test['ic50']=3.009075
    test.dropna(inplace=True)
    x=np.array(test.iloc[0,0:])
    print(x)

    #scale the input values
    scaled_x=scaler.transform([x])
    print(scaled_x)

    #predict drug encode
    drugenc=model.predict([scaled_x[0]])
    print(drugenc[0])

    #return the drugname
    drugname=tframe['Drug Name'].where(tframe['Drug Nameenc']==drugenc[0])
    drugname.dropna(inplace=True)
    print(drugname.iloc[0])
    
if __name__ == '__main__':
    file = sys.argv[1]
    main()