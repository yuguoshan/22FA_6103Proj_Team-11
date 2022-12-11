# /Users/medhaswetasen/Documents/GitHub/22FA_6103Proj_Team-11/EDA AND STATISTICAL TESTS
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
# BASIC MEASURES
print(df.head())
print(df.tail())
print(df.columns)
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df.isnull().sum().sum())
print(df.info())
print(df.describe())
#%%
# VISUALIZING THE DATA (EDA)
# BAR CHAR AND PIE CHARTS FOR ALL CATEGORICAL VARIABLES.
#%%
# JOB
figure, axis = plt.subplots(3,3)
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
# %%
# EDUCATION 
job_count = df['education'].value_counts()
job_count
plt.figure(figsize = (8, 5))
job_count.plot(kind = "bar")
plt.title("Type of Education Distribution")
#%%
# DEFAULT
job_count = df['default'].value_counts()
job_count
plt.figure(figsize = (8, 5))
job_count.plot(kind = "bar")
plt.title("Type of Default Distribution")
#%%
# HOUSING
job_count = df['housing'].value_counts()
job_count
plt.figure(figsize = (8, 5))
job_count.plot(kind = "bar")
plt.title("Type of Housing Distribution")
#%%
# LOAN
job_count = df['loan'].value_counts()
job_count
plt.figure(figsize = (8, 5))
job_count.plot(kind = "bar")
plt.title("Type of Loan Distribution")
#%%
# CONTACT
job_count = df['contact'].value_counts()
job_count
plt.figure(figsize = (8, 5))
job_count.plot(kind = "bar")
plt.title("Type of Contact Distribution")
#%%
# DAYS 
job_count = df['day_of_week'].value_counts()
job_count
plt.figure(figsize = (8, 5))
job_count.plot(kind = "bar")
plt.title("Type of Days Distribution")
#%%
# MONTH
job_count = df['month'].value_counts()
job_count
plt.figure(figsize = (8, 5))
job_count.plot(kind = "bar")
plt.title("Type of Months Distribution")
#%%
# POUTCOME
job_count = df['poutcome'].value_counts()
job_count
plt.figure(figsize = (8, 5))
job_count.plot(kind = "bar")
plt.title("Type of Poutcome Distribution")
#%%
# OUTCOME
job_count = df['y'].value_counts()
job_count
plt.figure(figsize = (8, 5))
job_count.plot(kind = "bar")
plt.title("Type of Outcome Distribution")
plt.tight_layout()
plt.show()
#%%
def pieChart(x_var,title):
    yesNo = df.groupby(x_var).size()
    yesNo.plot(kind='pie', title=title, figsize=[8,8],
          autopct=lambda p: '{:.2f}%({:.0f})'.format(p,(p/100)*yesNo.sum()))
    plt.show()

pieChart('y','Percentage of yes and no target(term deposit)in dataset')
#%%
# JOB
pieChart('job','Distribution of job in dataset')
df.poutcome.value_counts()
df.groupby('job').size()
#%%
# MARITAL
pieChart('marital','Distribution of marital in dataset')
df.poutcome.value_counts()
df.groupby('marital').size()
#%%
# EDUCATION
pieChart('education','Distribution of education in dataset')
df.poutcome.value_counts()
df.groupby('education').size()
#%%
# DEFAULT
pieChart('default','Distribution of default in dataset')
df.poutcome.value_counts()
df.groupby('default').size()
#%%
# HOUSING
pieChart('housing','Distribution of housing in dataset')
df.poutcome.value_counts()
df.groupby('housing').size()
#%%
# LOAN
pieChart('loan','Distribution of loan in dataset')
df.poutcome.value_counts()
df.groupby('loan').size()
#%%
# CONTACT
pieChart('contact','Distribution of contact in dataset')
df.poutcome.value_counts()
df.groupby('contact').size()
#%%
# DAY
pieChart('day_of_week','Distribution of day in dataset')
df.poutcome.value_counts()
df.groupby('day_of_week').size()
#%%
# MONTH
pieChart('month','Distribution of month in dataset')
df.poutcome.value_counts()
df.groupby('month').size()
#%%
# POUTCOME
pieChart('poutcome','Distribution of poutcome in dataset')
df.poutcome.value_counts()
df.groupby('poutcome').size()
#%%
# OUTCOME
pieChart('y','Distribution of outcome in dataset')
df.poutcome.value_counts()
df.groupby('y').size()
#%%
#%%
# HISTOGRAM AND DESITY PLOTS FOR ALL NUMERIC VARIABLE
#%%
sns.distplot(a=df.age, bins=40, color='green',
             hist_kws={"edgecolor": 'black'})
plt.title('Age Distribution')
plt.show()
# %%
sns.distplot(a=df.duration, bins=40, color='green',
             hist_kws={"edgecolor": 'black'})
plt.title('Duration Distribution')
plt.show()
# %%
sns.distplot(a=df.campaign, bins=40, color='green',
             hist_kws={"edgecolor": 'black'})
plt.title('Campaign Distribution')
plt.show()
# %%
sns.distplot(a=df.pdays, bins=40, color='green',
             hist_kws={"edgecolor": 'black'})
plt.title('Pdays Distribution')
plt.show()
# %%
sns.distplot(a=df.previous, bins=40, color='green',
             hist_kws={"edgecolor": 'black'})
plt.title('Previous Distribution')
plt.show()
#%%
sns.distplot(a=df['emp.var.rate'], bins=40, color='green',
             hist_kws={"edgecolor": 'black'})
plt.title('Emp.var.rate Distribution')
plt.show()
#%%
sns.distplot(a=df['cons.price.idx'], bins=40, color='green',
             hist_kws={"edgecolor": 'black'})
plt.title('Cons.price.idx Distribution')
plt.show()
#%%
sns.distplot(a=df['cons.conf.idx'], bins=40, color='green',
             hist_kws={"edgecolor": 'black'})
plt.title('Cons.conf.idx Distribution')
plt.show()
#%%
sns.distplot(a=df.euribor3m, bins=40, color='green',
             hist_kws={"edgecolor": 'black'})
plt.title('Euribor3m Distribution')
plt.show()
#%%
sns.distplot(a=df['nr.employed'], bins=40, color='green',
             hist_kws={"edgecolor": 'black'})
plt.title('Nr.employed Distribution')
plt.show()
#%%
#SCATTERPLOTS
#%%
numerical = ['age','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']
categorical = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','y']
#%%
# BOXPLOTS
for i in range (0,len(numerical)):
    sns.boxplot(data=df,x=numerical[i],y='y')
    plt.show()
# %%
# SCATTERPLOTS
for i in range (0,len(numerical)):
    for j in range (i+1,len(numerical)):
        sns.scatterplot(data=df,x=numerical[i],y=numerical[j],hue='y')
        plt.show()
# %%
# HEATMAPS
glue = df[['age','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']]
df_norm_col=(glue-glue.mean())/glue.std()
sns.heatmap(df_norm_col, cmap='viridis')
plt.show()
#%%
# SOMETHING TO LOOK AT (Dendrogram with heatmap and coloured leaves)
dfz= df[['age','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','y','balance']]
dfz['y'] = dfz['y'].apply(lambda x: 0 if x == 'no' else 1)
my_palette = dict(zip(dfz.y.unique(), ["orange","yellow"]))
row_colors = dfz.y.map(my_palette)
# plot
sns.clustermap(dfz, metric="correlation", method="single", cmap="Blues", standard_scale=1, row_colors=row_colors)
plt.show()
# %%
# STATISTICAL TESTS
# CHISQUARED TEST
for i in range (0,len(categorical)):
    for j in range (i+1,len(categorical)):
        crosstab, test_results, expected = rp.crosstab(df[categorical[i]], df[categorical[j]],
                                               test= "chi-square",
                                               expected_freqs= True,
                                               prop= "cell")
        print(crosstab)
        print(test_results)
        print(expected)
        fig = plt.figure(figsize=(6,6))
        sns.heatmap( crosstab, annot=True, cmap='Blues')
        plt.title('Chi-Square Test Results')
        plt.show()


# %%
# ONE-WAY ANOVA
for i in range (0,len(categorical)):
    for j in range (0,len(numerical)):
        model = ols("df[numerical[j]]~ df[categorical[i]]", data= df).fit()
        aov_table = sm.stats.anova_lm(model, typ=1)
        print(aov_table)
        print(model.summary())
        comp = mc.MultiComparison(df[numerical[j]], df[categorical[i]])
        post_hoc_res = comp.tukeyhsd()
        print(post_hoc_res.summary())
# %%
