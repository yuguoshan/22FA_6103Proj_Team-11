# /Users/medhaswetasen/Documents/GitHub/22FA_6103Proj_Team-11/Cleaning.py (python)
#%%
# IMPORTING NECESSARY LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import statsmodels.stats.multicomp as mc
import statsmodels.stats.outliers_influence as influence
import statsmodels.stats.diagnostic as diag
import statsmodels.stats.stattools as stattools
import statsmodels.stats.anova as anova
import statsmodels.stats.weightstats as weightstats
import statsmodels.stats.libqsturng as libqsturng
import statsmodels.stats.power as power
import statsmodels.stats.proportion as proportion
import statsmodels.stats.contingency_tables as contingency_tables
import statsmodels.stats.multitest as multitest
import statsmodels.stats.diagnostic as diagnostic
import statsmodels.stats.correlation_tools as correlation_tools
from statsmodels.formula.api import ols
import researchpy as rp
import scipy.stats as stats
#%%
# IMPORTING THE DATASET
inputFile = "bank copy.csv"
df = pd.read_csv(inputFile, sep=';')
#%%

print(df.isnull().sum())
print(df.isnull().sum().sum())
# %%
numerical = ['age','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']
categorical = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','y']
categorical_others = ['job','marital','housing']

for i in range (len(categorical_others)):
    for j in range (0,len(df[categorical_others[i]])):
        if df[categorical_others[i]][j] == 'unknown':
            df[categorical_others[i]][j] = 'others'
            
#%%
# JOB
job_count = df['job'].value_counts()
job_count
plt.figure(figsize = (8, 5))
job_count.plot(kind = "bar")
plt.title("Type of Job Distribution")
# %%
# MARITAL
job_count = df['marital'].value_counts()
job_count
plt.figure(figsize = (8, 5))
job_count.plot(kind = "bar")
plt.title("Type of Marital Distribution")            
#%%
# HOUSING
job_count = df['housing'].value_counts()
job_count
plt.figure(figsize = (8, 5))
job_count.plot(kind = "bar")
plt.title("Type of Housing Distribution")
#%%
# NOW CLEANING POUTCOME
for j in range (0,len(df["poutcome"])):
    if (df["poutcome"][j] == 'nonexistent'):
        if (df["default"][j] == 'yes'):
            df["poutcome"][j] = 'success'
        elif(df["default"][j] == 'no'):
            df["poutcome"][j] = 'success'
        elif(df["default"][j] == 'unknown'):
            df["poutcome"][j] = 'failure'
    
#%%
# POUTCOME
job_count = df['poutcome'].value_counts()
job_count
plt.figure(figsize = (8, 5))
job_count.plot(kind = "bar")
plt.title("Type of Poutcome Distribution")                
#%%  
categorical_no = ['loan','default']         
#%%
for i in range (len(categorical_no)):
    for j in range (0,len(df[categorical_no[i]])):
        if df[categorical_no[i]][j] == 'unknown':
            df[categorical_no[i]][j] = 'no'
#%%
# LOAN
job_count = df['loan'].value_counts()
job_count
plt.figure(figsize = (8, 5))
job_count.plot(kind = "bar")
plt.title("Type of Loan Distribution")
#%%
# DEFAULT
job_count = df['default'].value_counts()
job_count
plt.figure(figsize = (8, 5))
job_count.plot(kind = "bar")
plt.title("Type of Default Distribution")
#%%
df.to_csv("clean bank.csv")
# %%
