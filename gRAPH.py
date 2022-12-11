#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
inputFile = "primary.csv"
df = pd.read_csv(inputFile)

categorical = ['job','marital','education','default','housing','loan','contact','month','day','poutcome','y']
#%%
plt.figure(figsize=(50, 20))
for i in range (0,len(categorical)):
    ax = plt.subplot(3, 4, i+1)
    sns.countplot(data=df,x=categorical[i],ax=ax)
plt.tight_layout()
# %%
numerical = ['age','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','balance']
plt.figure(figsize=(50, 20))
for i in range (0,len(numerical)):
    ax = plt.subplot(3, 4, i+1)
    sns.distplot(a=df[numerical[i]], bins=40, color='green',
             hist_kws={"edgecolor": 'black'},ax=ax)
plt.title(numerical[i])
plt.tight_layout()
# %%
numerical = ['age','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','balance']
plt.figure(figsize=(50, 20))
for i in range (0,len(numerical)):
    ax = plt.subplot(3, 4, i+1)
    sns.boxplot(data=df,x=numerical[i],y='y',ax=ax)
plt.title(numerical[i])
plt.tight_layout()
# %%
plt.figure(figsize=(50, 20))
for i in range (0,len(numerical)):
    ax = plt.subplot(3, 4, i+1)
    sns.boxplot(data=df,x=numerical[i],y='y',ax=ax)
    plt.title(numerical[i])
plt.tight_layout()
#%%
for i in range (0,len(numerical)):
    for j in range (i+1,len(numerical)):
        ax = plt.subplot(7, 7,i+j+1)
        sns.scatterplot(data=df,x=numerical[i],y=numerical[j],hue='y',ax=ax)
plt.tight_layout()        
# %%
glue = df.pivot('age','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','balance')
#df_norm_col=(glue-glue.mean())/glue.std()
sns.heatmap(glue, annot=True, fmt=".1f")
plt.show()
# %%
