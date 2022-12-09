#%%
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
#%%
inputFile = "bank.csv"
df = pd.read_csv(inputFile, sep=';')
    
#%%
numerical = ['age','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']
categorical = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','y']
categorical_independent = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for i in categorical:
    df=df[df[i]!= "unknown"]
    
one_hot_encoded_data = pd.get_dummies(df, columns = categorical_independent)
one_hot_encoded_data.dtypes
df['target'] = df['y'].apply(lambda row: 1 if row == 'yes' else 0) 
one_hot_encoded_data.to_csv('file1.csv')


# %%
x = one_hot_encoded_data.drop("y", axis='columns')
y = df['target']
X_train, X_test, y_train, y_test= train_test_split(x, y, test_size=0.2,random_state=32)
# %%
# decision tree
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(X_train, y_train)
dtcprediction = dtc.predict(X_test)
print(accuracy_score(y_test, dtcprediction))
print(confusion_matrix(y_test, dtcprediction))
print(classification_report(y_test, dtcprediction))
# %%
# random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfcpredictions = rfc.predict(X_test)
print(accuracy_score(y_test, rfcpredictions))
print(confusion_matrix(y_test, rfcpredictions))
print(classification_report(y_test, rfcpredictions))

# %%
#  SVM - Support Vector Machines
from sklearn.svm import SVC, LinearSVC
svc_linear = LinearSVC()
svc_linear.fit(X_train, y_train)
svc_linearpredictions = svc_linear.predict(X_test)
print(accuracy_score(y_test, svc_linearpredictions))
print(confusion_matrix(y_test, svc_linearpredictions))
print(classification_report(y_test, svc_linearpredictions))

svc= SVC()
svc.fit(X_train, y_train)
svcpredictions = svc.predict(X_test)
print(accuracy_score(y_test, svcpredictions))
print(confusion_matrix(y_test, svcpredictions))
print(classification_report(y_test, svcpredictions))
# %%
