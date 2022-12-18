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
inputFile = "encoded_unbalanced.csv"
df = pd.read_csv(inputFile)
    

# %%
x = df.drop("y", axis='columns')
y = df['y']
X_train, X_test, y_train, y_test= train_test_split(x, y, test_size=0.2,random_state=1234)
# %%
# decision tree
# feature selection
from matplotlib import pyplot
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
importance = dtc.feature_importances_
features = [] 
for i,v in enumerate(importance):
    if v >0.01:
        print(f"Feature {i} variable {df.columns[i]} score {v}")
        features.append(df.columns[i])
print(features)
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
# %%
## decision tree
# filter the variables
x_select = df[features].drop("y",axis='columns')
X_train_f, X_test_f, y_train_f, y_test_f= train_test_split(x_select, y, test_size=0.2,random_state=1234)
dtc = DecisionTreeClassifier()
dtc.fit(X_train_f, y_train_f)
dtcprediction = dtc.predict(X_test_f)
print(accuracy_score(y_test_f, dtcprediction))
print(confusion_matrix(y_test_f, dtcprediction))
print(classification_report(y_test_f, dtcprediction))


# %%
# random forest
# feature selection
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
importance_rfc = rfc.feature_importances_
features_rfc = [] 
for i,v in enumerate(importance_rfc):
    if v >0.01:
        print(f"Feature {i} variable {df.columns[i]} score {v}")
        features_rfc.append(df.columns[i])
print(features_rfc)
pyplot.bar([x for x in range(len(importance_rfc))], importance_rfc)
pyplot.show()

# %%
# random forest
# filter the variables
x_select_rfc = df[features_rfc].drop("y",axis='columns')
X_train_rfc, X_test_rfc, y_train_rfc, y_test_rfc= train_test_split(x_select_rfc, y, test_size=0.2,random_state=1234)
rfc = RandomForestClassifier()
rfc.fit(X_train_rfc, y_train_rfc)
rfcpredictions = rfc.predict(X_test_rfc)
print(accuracy_score(y_test_rfc, rfcpredictions ))
print(confusion_matrix(y_test_rfc, rfcpredictions ))
print(classification_report(y_test_rfc, rfcpredictions ))






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
