#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import imblearn
#%%
from sklearn.model_selection import train_test_split
#Reading the dataset 
filename = 'encoded_unbalanced.csv'
data = pd.read_csv(filename)
#dropping y to extract x variables 
x = data.drop(['y'],axis=1)
#y variables
y= data['y']
#splitting the dataset 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
#%%
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(x_train, y_train)
test_sx, test_sy = sm.fit_resample(x_test, y_test)
#printing x and y values 
np.bincount(y_res)

# %%
train_balanced = pd.concat([X_res, y_res], axis=1)
train_unbalanced = pd.concat([x_train, y_train], axis=1)

test_unbalanced = pd.concat([x_test, y_test], axis=1)
test_balanced = pd.concat([test_sx, test_sy], axis=1)
# %%
print("Before Smote")
print(f"for training : {np.bincount(y_train)}")
print(f"for testing : {np.bincount(y_test)}")
print("After smote")
print(f"for training : {np.bincount(y_res)}")
print(f"for testing : {np.bincount(test_sy)}")
# %%

#saving all the models 

train_balanced.to_csv('Dataset/train_balanced.csv',index=False)
train_unbalanced.to_csv('Dataset/train_unbalanced.csv',index=False)
test_unbalanced.to_csv('Dataset/test_unbalanced.csv',index=False)
test_balanced.to_csv('Dataset/test_balanced.csv',index=False)

# %%
