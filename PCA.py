#
#%%
# IMPORTING THE DATASET
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from collections import Counter
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
inputFile = "unbalanced_bin.csv"
df= pd.read_csv(inputFile)
print(df.shape)
from sklearn.preprocessing import StandardScaler
#%%

x_pca = df.drop(columns=['y'])
x_pca = StandardScaler().fit_transform(x_pca)
from sklearn.decomposition import PCA

pca = PCA(n_components=20)

principalComponents = pca.fit_transform(x_pca)
principalDF = pd.DataFrame(data = principalComponents)
print(pca.components_)
print(pca.explained_variance_ratio_)
# Concatenation of dataframes
df_all = pd.concat([principalDF, df], axis=1)
plt.figure(figsize=(15,5))
sns.heatmap(df_all.corr(), annot=True, fmt = ".2f", cmap = "coolwarm")
# %%
