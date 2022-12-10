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
