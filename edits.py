#%%
import pandas as pd
inputFile = "bankfull.csv"
df = pd.read_csv(inputFile, sep=';')
df.head()
#%%
df.shape
#%%
df.dtypes
#%%
import matplotlib.pyplot as plt
import seaborn as sns
#%%
sns.set_theme(style="whitegrid")
#%%
job_count = df['job'].value_counts()
job_count
#%%
plt.figure(figsize = (8, 5))
job_count.plot(kind = "bar")
plt.title("Type of Job Distribution")
#%%
default_count = df['default'].value_counts()
default_count
#%%
plt.figure(figsize = (8, 5))
default_count.plot(kind='bar').set(title='Default Column Distribution')
#%%
marital_count = df['marital'].value_counts()
marital_count
#%%
plt.figure(figsize = (8, 5))
marital_count.plot(kind = "bar").set(title = "Merital Distribution")
#%%
loan_count = df['loan'].value_counts()
loan_count
#%%

plt.figure(figsize = (8, 5))
loan_count.plot(kind = "bar").set(title = "Loan Distribution")
#%%
housing_count = df['housing'].value_counts()
housing_count
#%%
plt.figure(figsize = (8, 5))
housing_count.plot(kind = "bar").set(title = "Housing Loan Distribution")
#%%
education_count = df['education'].value_counts()
education_count
#%%
plt.figure(figsize = (8, 5))
education_count.plot(kind = "bar").set(title = "Education Column Distribution")
#%%
contact_count = df['contact'].value_counts()
contact_count
#%%
plt.figure(figsize = (8, 5))
contact_count.plot(kind = "bar").set(title = "Contact Column Distribution")
#%%
month_count = df['month'].value_counts()
month_count
#%%
plt.figure(figsize = (8, 5))
month_count.plot(kind = "bar").set(title = "Month Data Distribution")
#%%
plt.figure(figsize = (8, 5))
df['pdays'].hist(bins = 50)
#%%
plt.figure(figsize = (8, 5))
df[df['pdays'] > 0]['pdays'].hist(bins=50)
#%%
target_count = df['y'].value_counts()
target_count
#%%
plt.figure(figsize = (8, 5))
target_count.plot(kind = "bar").set(title = "Target Distribution")
#%%
df[df['y'] == 'yes'].hist(figsize = (20,20))
plt.title('Client has subscribed a term deposite')
#%%
df[df['y'] == 'no'].hist(figsize = (20,20))
plt.title('Client has not subscribed a term deposite')
#%%
df.head(10)
#%%
df['is_default'] = df['default'].apply(lambda row: 1 if row == 'yes' else 0)
#%%
df[['default','is_default']].tail(10)
#%%
df['is_housing'] = df['housing'].apply(lambda row: 1 if row == 'yes' else 0)
df[['housing','is_housing']].tail(10)
#%%
df['is_loan'] = df['loan'].apply(lambda row: 1 if row == 'yes' else 0)
df[['loan', 'is_loan']].tail(10)
#%%
df['target'] = df['y'].apply(lambda row: 1 if row == 'yes' else 0)
df[['y', 'target']].tail(10)
#%%
marital_dummies = pd.get_dummies(df['marital'], prefix = 'marital')
marital_dummies.tail()
#%%
pd.concat([df['marital'], marital_dummies], axis=1).head(n=10)
#%%
marital_dummies.drop('marital_divorced', axis=1, inplace=True)
marital_dummies.head()
#%%
df = pd.concat([df, marital_dummies], axis=1)
df.head()
#%%
job_dummies = pd.get_dummies(df['job'], prefix = 'job')
job_dummies.tail()
#%%
job_dummies.drop('job_unknown', axis=1, inplace=True)
#%%
df = pd.concat([df, job_dummies], axis=1)
df.head()
#%%
education_dummies = pd.get_dummies(df['education'], prefix = 'education')
education_dummies.tail()
#%%
education_dummies.drop('education_unknown', axis=1, inplace=True)
education_dummies.tail()
#%%
df = pd.concat([df, education_dummies], axis=1)
df.head()
#%%
contact_dummies = pd.get_dummies(df['contact'], prefix = 'contact')
contact_dummies.tail()
#%%
contact_dummies.drop('contact_unknown', axis=1, inplace=True)
contact_dummies.tail()
#%%
df = pd.concat([df, contact_dummies], axis=1)
df.head()
#%%
poutcome_dummies = pd.get_dummies(df['poutcome'], prefix = 'poutcome')
poutcome_dummies.tail()
#%%
poutcome_dummies.drop('poutcome_unknown', axis=1, inplace=True)
poutcome_dummies.tail()
#%%
df = pd.concat([df, poutcome_dummies], axis=1)
df.head()
#%%
months = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec': 12}
df['month'] = df['month'].map(months)
df['month'].head()
#%%
df[df['pdays'] == -1]['pdays'].count()
#%%
df['was_contacted'] = df['pdays'].apply(lambda row: 0 if row == -1 else 1)
df[['pdays','was_contacted']].head()
#%%
df.drop(['job', 'education', 'marital', 'default', 'housing', 'loan', 'contact', 'pdays', 'poutcome', 'y'], axis=1, inplace=True)
#%%
df.dtypes
#%%
df.head(10)
#%%
#The axis=1 argument drop columns
X = df.drop('target', axis=1)
y = df['target']
#%%
X.shape
#%%
y.shape
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 32)
#%%
X_train.shape
#%%
y_train.shape
#%%
X_test.shape
#%%
y_test.shape
#%%
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
#%%
model.fit(X_train, y_train)
#%%
y_pred = model.predict(X_test)
#%%
print("Predicted value: ", y_pred[:10])
print("Actual value: ", y_test[:10])
#%%
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_pred = y_pred, y_true = y_test)
print(f'Accuracy of the model Logistic Regression is {accuracy*100:.2f}%')
#%%
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfcpredictions = rfc.predict(X_test)
print("Predicted value: ", rfcpredictions[:10])
print("Actual value: ", y_test[:10])
#%%
accuracy = accuracy_score(y_pred = rfcpredictions, y_true = y_test)
print(f'Accuracy of the Random Forest Classifier model is {accuracy*100:.2f}%')
#%%
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
svcpredictions = svc.predict(X_test)
print("Predicted value: ", svcpredictions[:10])
print("Actual value: ", y_test[:10])
#%%
accuracy = accuracy_score(y_pred = svcpredictions, y_true = y_test)
print(f'Accuracy of the SVC model is {accuracy*100:.2f}%')
#%%
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(X_train, y_train)
dtcprediction = dtc.predict(X_test)
print("Predicted value: ", dtcprediction[:10])
print("Actual value: ", y_test[:10])
#%%
accuracy = accuracy_score(y_pred = dtcprediction, y_true = y_test)
print(f'Accuracy of the Decision Tree Classifier model is {accuracy*100:.2f}%')