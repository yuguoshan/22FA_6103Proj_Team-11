#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
#%%
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report

# %%
#Reading the dataset 
filename = 'encoded_unbalanced.csv'
data = pd.read_csv(filename)
#dropping y to extract x variables 
x = data.drop(['y'],axis=1)
#y variables
y= data['y']
#splitting the dataset 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
# %%
#modelling
from sklearn.naive_bayes import GaussianNB
modelNB = GaussianNB()
modelNB.fit(x_train,y_train)

# %%
print(f"Model score is {modelNB.score(x_test,y_test)}")
#%%

def modelProbability(prediction0,prediction1,y):
    plt.figure(figsize=(15,7))
    #plt.hist(prediction1[y==0], bins=50, label='No_pred1', alpha=0.7, color='g')
    #plt.hist(prediction0[y==0], bins=50, label='No_pred0')
    plt.hist(prediction0[y==1], bins=50, label='Yes_pred0', alpha=0.7, color='r')
    plt.hist(prediction1[y==1], bins=50, label='Yes_pred1', alpha=0.7, color='y')
    plt.xlabel('Probability of being Positive Class', fontsize=25)
    plt.ylabel('Number of records in each bucket', fontsize=25)
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', labelsize=25, pad=5)
    plt.show() 
pred1=modelNB.predict_proba(x_test)[:,0]
pred2 = modelNB.predict_proba(x_test)[:,1]
modelProbability(pred1,pred2,y_test)

# %%
#Applying Naive Bayes on upsampled dataset 
#Reading the training and testing dataset 

balanced_train = pd.read_csv('Dataset\\train_balanced.csv')
unbalanced_train = pd.read_csv('Dataset\\train_unbalanced.csv')
balanced_test = pd.read_csv('Dataset\\test_balanced.csv')
unbalanced_test = pd.read_csv('Dataset\\test_unbalanced.csv')

# %%

#modelling
def xAndY(dataset):
    x = dataset.drop(['y'],axis=1)
    y = dataset['y']
    return x,y
def evaluateNB(dataset):
    x_train,y_train = xAndY(dataset)
    model_temp = GaussianNB()
    model_temp.fit(x_train,y_train)
    return model_temp
def modelEvaluation(model,x,y):
    print('test set evaluation: ')
    y_pred = model.predict(x)
    print(accuracy_score(y, y_pred))
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))
    
          
    
model_balanced = evaluateNB(balanced_train)
model_unbalanced = evaluateNB(unbalanced_train)
#checking it on the unbalanced test dataset only 
x_test,y_test = xAndY(unbalanced_test)
print(f"Training : balanced \n Testing : unbalanced")
modelEvaluation(model_balanced,x_test,y_test)
   
print(f"Training : unbalanced \n Testing : unbalanced")
modelEvaluation(model_unbalanced, x_test,  y_test) 
# %%
# balanced Testing dataset 
#checking it on the unbalanced test dataset only 
x_test,y_test = xAndY(balanced_test)
print(f"Training : balanced \n Testing : balanced")
modelEvaluation(model_balanced,x_test,y_test)
   
# %%
