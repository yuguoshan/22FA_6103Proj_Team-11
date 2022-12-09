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
inputFile = "encoded_unbalanced.csv"
df= pd.read_csv(inputFile)
print(df.shape)
#%%
from sklearn.model_selection import train_test_split
X = df.drop(columns=['y'])
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1234)

#%%
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
rfe_model = RFE(LogisticRegression(solver='lbfgs', max_iter=1000), step= 25)
rfe_model = rfe_model.fit(X_train,y_train)

# feature selection
print(rfe_model.support_)
print(rfe_model.ranking_)
#%%
selected_columns = X_train.columns[rfe_model.support_]
print(selected_columns.tolist())
#%%
X_train_final = X_train[selected_columns.tolist()]
y_train_final = y_train
X_test_final = X_test[selected_columns.tolist()]
y_test_final = y_test

#%%
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logreg = LogisticRegression()
logreg.fit(X_train_final, y_train_final)
#%%
y_pred = logreg.predict(X_test_final)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test_final, y_test_final)))

#%%
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


#%%
print('X_train_final type',type(X_train_final))
print('X_train_final shape',X_train_final.shape)
print('X_test_final type',type(X_test_final))
print('X_test_final shape',X_test_final.shape)
print('y_train_final type',type(y_train_final))
print('y_train_final shape',y_train_final.shape)
print('y_test_final type',type(y_test_final))
print('y_test_final shape',y_test_final.shape)

print("\nReady to continue.")

#%%
# Logit
from sklearn.linear_model import LogisticRegression

logit = LogisticRegression()  # instantiate
logit.fit(X_train_final, y_train_final)
print('Logit model accuracy (with the test set):', logit.score(X_test_final, y_test_final))
print('Logit model accuracy (with the train set):', logit.score(X_train_final, y_train_final))

print("\nReady to continue.")

#%%
print(logit.predict(X_test_final))

print("\nReady to continue.")

#%%
print(logit.predict_proba(X_train_final[:8]))
print(logit.predict_proba(X_test_final[:8]))

print("\nReady to continue.")

#%%
test = logit.predict_proba(X_test_final)
type(test)

print("\nReady to continue.")

#%%
#
# Write a function to change the proba to 0 and 1 base on a cut off value.
# 
#%%
cut_off = 0.20
predictions = (logit.predict_proba(X_test_final)[:,1]>cut_off).astype(int)
print(predictions)

print("\nReady to continue.")

# print("\nReady to continue.")


#%%
# Classification Report
#
from sklearn.metrics import classification_report
y_true, y_pred = y_test_final, logit.predict(X_test_final)
print(classification_report(y_true, y_pred))

#                         predicted 
#                   0                  1
# Actual 0   True Negative  TN      False Positive FP
# Actual 1   False Negative FN      True Positive  TP
# 
# Accuracy    = (TP + TN) / Total
# Precision   = TP / (TP + FP)
# Recall rate = TP / (TP + FN) = Sensitivity
# Specificity = TN / (TN + FP)
# F1_score is the "harmonic mean" of precision and recall
#          F1 = 2 (precision)(recall)/(precision + recall)

print("\nReady to continue.")

#%%
# Precision-Recall vs Threshold

y_pred=logit.predict(X_test_final)

y_pred_probs=logit.predict_proba(X_test_final) 
# probs_y is a 2-D array of probability of being labeled as 0 (first 
# column of array) vs 1 (2nd column in array)

from sklearn import metrics
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test_final, y_pred_probs[:, 1]) 
#retrieve probability of being 1(in second column of probs_y)
pr_auc = metrics.auc(recall, precision)

plt.title("Precision-Recall vs Threshold Chart")
plt.plot(thresholds, precision[: -1], "b--", label="Precision")
plt.plot(thresholds, recall[: -1], "r--", label="Recall")
plt.ylabel("Precision, Recall")
plt.xlabel("Threshold")
plt.legend(loc="lower left")
plt.ylim([0,1])

print("\nReady to continue.")

#%%
# Receiver Operator Characteristics (ROC)
# Area Under the Curve (AUC)
from sklearn.metrics import roc_auc_score, roc_curve

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test_final))]
# predict probabilities
lr_probs = logit.predict_proba(X_test_final)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test_final, ns_probs)
lr_auc = roc_auc_score(y_test_final, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test_final, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test_final, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# aXis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()


# %%
#%%
# K-Nearest-Neighbor KNN   on  admissions data
# number of neighbors
mrroger = 7

print("\nReady to continue.")

#%%
# KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=mrroger) # instantiate with n value given
knn.fit(X,y)
y_pred = knn.predict(X)
y_pred = knn.predict_proba(X)
print(y_pred)
print(knn.score(X,y))

print("\nReady to continue.")

#%%
# 2-KNN algorithm
# The better way
# from sklearn.neighbors import KNeighborsClassifier
knn_split = KNeighborsClassifier(n_neighbors=mrroger) # instantiate with n value given
knn_split.fit(X_train,y_train)
ytest_pred = knn_split.predict(X_test)
ytest_pred
print(knn_split.score(X_test,y_test))

# Try different n values

print("\nReady to continue.")

#%%
# 3-KNN algorithm
# The best way
from sklearn.neighbors import KNeighborsClassifier
knn_cv = KNeighborsClassifier(n_neighbors=mrroger) # instantiate with n value given

from sklearn.model_selection import cross_val_score
cv_results = cross_val_score(knn_cv, X, y, cv=10)
print(cv_results) 
print(np.mean(cv_results)) 

print("\nReady to continue.")

#%%
# 4-KNN algorithm
# Scale first? better or not?

# Re-do our darta with scale on X
from sklearn.preprocessing import scale
xsadmit = pd.DataFrame( scale(X), columns=X.columns )  # reminder: X = dfadmit[['gre', 'gpa', 'rank']]
# Note that scale( ) coerce the object from pd.dataframe to np.array  
# Need to reconstruct the pandas df with column names
# xsadmit.rank = X.rank
ysadmit = y.copy()  # no need to scale y, but make a true copy / deep copy to be safe

print("\nReady to continue.")

#%%
# from sklearn.neighbors import KNeighborsClassifier
knn_scv = KNeighborsClassifier(n_neighbors=mrroger) # instantiate with n value given

# from sklearn.model_selection import cross_val_score
scv_results = cross_val_score(knn_scv, xsadmit, ysadmit, cv=5)
print(scv_results) 
print(np.mean(scv_results)) 

print("\nReady to continue.")
