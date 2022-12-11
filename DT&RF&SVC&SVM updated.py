#%%
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
from matplotlib import pyplot
#%%
balanced_train = pd.read_csv('train_balanced.csv').drop(['day'],axis=1)
unbalanced_train = pd.read_csv('train_unbalanced.csv').drop(['day'],axis=1)
unbalanced_test = pd.read_csv('test_unbalanced.csv').drop(['day'],axis=1)
    
# %%
# split the dataset 
def xAndY(dataset):
    x = dataset.drop(['y'],axis=1)
    y = dataset['y']
    return x,y

x_train_ba,y_train_ba = xAndY(balanced_train)
x_train_un,y_train_un = xAndY(unbalanced_train)
x_test_un,y_test_un = xAndY(unbalanced_test)


# %%
#  SVC balance check on unbalance test
from sklearn.svm import SVC, LinearSVC
svc_linear_ba = LinearSVC()
svc_linear_ba.fit(x_train_ba,y_train_ba)
svc_linear_bapredictions = svc_linear_ba.predict(x_test_un)
print(accuracy_score(y_test_un, svc_linear_bapredictions))
print(confusion_matrix(y_test_un, svc_linear_bapredictions))
print(classification_report(y_test_un, svc_linear_bapredictions))

# %%
#  SVC unbalance check on unbalance test
svc_linear = LinearSVC()
svc_linear.fit(x_test_un,y_test_un)
svc_linearpredictions = svc_linear.predict(x_test_un)
print(accuracy_score(y_test_un, svc_linearpredictions))
print(confusion_matrix(y_test_un, svc_linearpredictions))
print(classification_report(y_test_un, svc_linearpredictions))


# %%
#  SVM - Support Vector Machines balance check on unbalance test
svc= SVC()
svc.fit(x_train_ba,y_train_ba)
svcpredictions = svc.predict(x_test_un)
print(accuracy_score(y_test_un, svcpredictions))
print(confusion_matrix(y_test_un, svcpredictions))
print(classification_report(y_test_un, svcpredictions))

# %%
#  SVM - Support Vector Machines unbalance check on unbalance test
svc_un= SVC()
svc_un.fit(x_train_un,y_train_un)
svcpredictions = svc_un.predict(x_test_un)
print(accuracy_score(y_test_un, svcpredictions))
print(confusion_matrix(y_test_un, svcpredictions))
print(classification_report(y_test_un, svcpredictions))

# %%
# decision tree balance check on unbalance test
# feature selection
dtc = DecisionTreeClassifier()
dtc.fit(x_train_ba, y_train_ba)
importance = dtc.feature_importances_
features = [] 
for i,v in enumerate(importance):
    if v >0.01:
        print(f"Feature {i} variable {balanced_train.columns[i]} score {v}")
        features.append(balanced_train.columns[i])
print(features)
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
# %%
## decision tree balance check on unbalance test
# filter the variables
x_train_ba,y_train_ba = balanced_train[features],balanced_train['y']
x_test_un,y_test_un = unbalanced_test[features],unbalanced_test['y']

dtc = DecisionTreeClassifier()
dtc.fit(x_train_ba,y_train_ba )
dtcprediction = dtc.predict(x_test_un)
print(accuracy_score(y_test_un, dtcprediction))
print(confusion_matrix(y_test_un, dtcprediction))
print(classification_report(y_test_un, dtcprediction))

#feature_names = features
#target_names = list("y")
#plot_tree(dtc, feature_names=feature_names, class_names=target_names,filled=True)
# %%
# decision tree unbalance check on unbalance test
# feature selection
dtc = DecisionTreeClassifier()
dtc.fit(x_train_un, y_train_un)
importance = dtc.feature_importances_
features = [] 
for i,v in enumerate(importance):
    if v >0.01:
        print(f"Feature {i} variable {unbalanced_train.columns[i]} score {v}")
        features.append(unbalanced_train.columns[i])
print(features)
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
# %%
## decision tree unbalance check on unbalance test
# filter the variables
x_train_un,y_train_un = unbalanced_train[features],unbalanced_train['y']
x_test_un,y_test_un = unbalanced_test[features],unbalanced_test['y']

dtc = DecisionTreeClassifier()
dtc.fit(x_train_un,y_train_un )
dtcprediction = dtc.predict(x_test_un)
print(accuracy_score(y_test_un, dtcprediction))
print(confusion_matrix(y_test_un, dtcprediction))
print(classification_report(y_test_un, dtcprediction))



# %%
# random forest balance check on unbalance test
# feature selection
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train_ba, y_train_ba)
importance_rfc = rfc.feature_importances_
features_rfc = [] 
for i,v in enumerate(importance_rfc):
    if v >0.01:
        print(f"Feature {i} variable {balanced_train.columns[i]} score {v}")
        features_rfc.append(balanced_train.columns[i])
print(features_rfc)
pyplot.bar([x for x in range(len(importance_rfc))], importance_rfc)
pyplot.show()

# %%
# random forest balance check on unbalance test
# filter the variables
x_train_ba,y_train_ba = balanced_train[features],balanced_train['y']
x_test_un,y_test_un = unbalanced_test[features],unbalanced_test['y']

rfc = RandomForestClassifier()
rfc.fit(x_train_ba,y_train_ba)
rfcpredictions = rfc.predict(x_test_un)
print(accuracy_score(y_test_un, rfcpredictions ))
print(confusion_matrix(y_test_un, rfcpredictions ))
print(classification_report(y_test_un, rfcpredictions ))

# %%
# random forest unbalance check on unbalance test
# feature selection
rfc = RandomForestClassifier()
rfc.fit(x_train_un, y_train_un)
importance_rfc = rfc.feature_importances_
features_rfc = [] 
for i,v in enumerate(importance_rfc):
    if v >0.01:
        print(f"Feature {i} variable {unbalanced_train.columns[i]} score {v}")
        features_rfc.append(unbalanced_train.columns[i])
print(features_rfc)
pyplot.bar([x for x in range(len(importance_rfc))], importance_rfc)
pyplot.show()

# %%
# random forest unbalance check on unbalance test
# filter the variables
x_train_un,y_train_un = unbalanced_train[features],unbalanced_train['y']
x_test_un,y_test_un = unbalanced_test[features],unbalanced_test['y']

rfc = RandomForestClassifier()
rfc.fit(x_train_un,y_train_un)
rfcpredictions = rfc.predict(x_test_un)
print(accuracy_score(y_test_un, rfcpredictions ))
print(confusion_matrix(y_test_un, rfcpredictions ))
print(classification_report(y_test_un, rfcpredictions ))



