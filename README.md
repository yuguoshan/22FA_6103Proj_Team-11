# 22FA_6103Proj_Team-11

Hi, We are Team 11 of Data Mining Project. 

Our team members are:

1. Anjali Mudgal

2. Guoshan Yu

3. Medhasweta Sen

     AN ANALYSIS OF PORTUGUESE BANK MARKETING DATA code{white-space: pre-wrap;} span.smallcaps{font-variant: small-caps;} div.columns{display: flex; gap: min(4vw, 1.5em);} div.column{flex: auto; overflow-x: auto;} div.hanging-indent{margin-left: 1.5em; text-indent: -1.5em;} ul.task-list{list-style: none;} ul.task-list li input\[type="checkbox"\] { width: 0.8em; margin: 0 0.8em 0.2em -1.6em; vertical-align: middle; }     html{ scroll-behavior: smooth; }

Table of contents
-----------------

*   [1 INTRODUCTION](#introduction)
    *   [1.1 The Data Set](#the-data-set)
    *   [1.2 The SMART Questions](#the-smart-questions)
    *   [1.3 Importing the dataset](#importing-the-dataset)
    *   [1.4 Basic Information about the data](#basic-information-about-the-data)
*   [2 Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    *   [2.1 Distribution of y(target) variable](#distribution-of-ytarget-variable)
    *   [2.2 Missing values and Outliers](#missing-values-and-outliers)
        *   [2.2.1 Education](#education)
        *   [2.2.2 Contact](#contact)
        *   [2.2.3 Poutcome](#poutcome)
    *   [2.3 Outliers](#outliers)
*   [3 Data Cleaning](#data-cleaning)
    *   [3.1 Dropping the irrelavant columns and missing values](#dropping-the-irrelavant-columns-and-missing-values)
    *   [3.2 Outlier removal](#outlier-removal)
        *   [3.2.1 _Balance - Outliers_](#balance---outliers)
        *   [3.2.2 _Duration - Outliers_](#duration---outliers)
*   [4 Data Visualization](#data-visualization)
    *   [4.1 SMART Question 1 :](#smart-question-1)
        *   [4.1.1 Number of calls versus Duration and affect on subscription](#number-of-calls-versus-duration-and-affect-on-subscription)
    *   [4.2 Month wise subscription](#month-wise-subscription)
        *   [4.2.1 SMART Question 7: How are the likelihood of subscriptions affected by social and economic factors?](#smart-question-7-how-are-the-likelihood-of-subscriptions-affected-by-social-and-economic-factors)
        *   [4.2.2 SMART Question 2](#smart-question-2)
        *   [4.2.3 Loan](#loan)
        *   [4.2.4 Age](#age)
        *   [4.2.5 Job](#job)
        *   [4.2.6 Balance](#balance)
*   [5 Data Encoding](#data-encoding)
    *   [5.1 One Hot Encoding](#one-hot-encoding)
    *   [5.2 Sin - Cos encoding](#sin---cos-encoding)
    *   [5.3 Dropping unnecessary columns irrelevant for modelling](#dropping-unnecessary-columns-irrelevant-for-modelling)
*   [6 Data Modeling](#data-modeling)
    *   [6.1 Splitting our Dataset](#splitting-our-dataset)
    *   [6.2 Balancing Our Dataset](#balancing-our-dataset)
*   [7 Scaling numeric variables](#scaling-numeric-variables)
*   [8 Logistic Regression](#logistic-regression)
*   [9 Balanced Dataset](#balanced-dataset)
    *   [9.0.1 Deciding cut off value for logistic regression - Unbalance](#deciding-cut-off-value-for-logistic-regression---unbalance)
    *   [9.0.2 Smart Question 5: The optimal cut off value for classification of our imbalance dataset.](#smart-question-5-the-optimal-cut-off-value-for-classification-of-our-imbalance-dataset.)
    *   [9.0.3 SMART Question 2: Since the dataset is imbalanced, will down sampling/up sampling or other techniques improve upon the accuracy of models.](#smart-question-2-since-the-dataset-is-imbalanced-will-down-samplingup-sampling-or-other-techniques-improve-upon-the-accuracy-of-models.)
*   [10 Decision Tree](#decision-tree)
    *   [10.1 Feature Selection](#feature-selection)
    *   [10.2 Hyperparameter tuning](#hyperparameter-tuning)
*   [11 Random Forest](#random-forest)
    *   [11.1 Feature Selection](#feature-selection-1)
    *   [11.2 Hyperparameter Tuning](#hyperparameter-tuning-1)
*   [12 Linear SVC](#linear-svc)
*   [13 SVC](#svc)
*   [14 Naive Bayes](#naive-bayes)
    *   [14.1 For balanced](#for-balanced)
*   [15 KNN](#knn)
*   [16 ROC -AUC Curve](#roc--auc-curve)
*   [17 Precision Recall Curve](#precision-recall-curve)
*   [18 Summary](#summary)
*   [19 Conclusion](#conclusion)
*   [20 Reference](#reference)

AN ANALYSIS OF PORTUGUESE BANK MARKETING DATA
=============================================

The George Washington University (DATS 6103: An Introduction to Data Mining)

Author

TEAM 11: Anjali Mudgal, Guoshan Yu, Medhasweta Sen

Published

December 20, 2022

1 INTRODUCTION
==============

Bank marketing is the practice of attracting and acquiring new customers through traditional media and digital media strategies. The use of these media strategies helps determine what kind of customer is attracted to a certain institutions. This also includes different banking institutions purposefully using different strategies to attract the type of customer they want to do business with.

Marketing has evolved from a communication role to a revenue generating role. The consumer has evolved from being a passive recipient of marketing messages to an active participant in the marketing process. Technology has evolved from being a means of communication to a means of data collection and analysis. Data analytics has evolved from being a means of understanding the consumer to a means of understanding the consumer and the institution.

Bank marketing strategy is increasingly focused on digital channels, including social media, video, search and connected TV. As bank and credit union marketers strive to promote brand awareness, they need a new way to assess channel ROI and more accurate data to enable personalized offers. Add to that the growing importance of purpose-driven marketing.

The relentless pace of digitization is disrupting not only the established order in banking, but bank marketing strategies. Marketers at both traditional institutions and digital disruptors are feeling the pressure.

Just as bank marketers begin to master one channel, consumers move to another. Many now toggle between devices on a seemingly infinite number of platforms, making it harder than ever for marketers to pin down the right consumers at the right time in the right place.

![](expected-marketing-budget-changes-by-channel.png)

1.1 The Data Set
----------------

The data set used in this analysis is from a Portuguese bank. The data set contains 41,188 observations and 21 variables. The variables include the following:

1.  *   age (numeric)
2.  *   job : type of job (categorical: ‘admin.’,‘blue-collar’,‘entrepreneur’,‘housemaid’,‘management’,‘retired’,‘self-employed’,‘services’,‘student’,‘technician’,‘unemployed’,‘unknown’)
3.  *   marital : marital status (categorical: ‘divorced’,‘married’,‘single’,‘unknown’; note: ‘divorced’ means divorced or widowed)
4.  *   education (categorical: ‘basic.4y’,‘basic.6y’,‘basic.9y’,‘high.school’,‘illiterate’,‘professional.course’,‘university.degree’,‘unknown’)
5.  *   default: has credit in default? (categorical: ‘no’,‘yes’,‘unknown’)
6.  *   housing: has housing loan? (categorical: ‘no’,‘yes’,‘unknown’)
7.  *   loan: has personal loan? (categorical: ‘no’,‘yes’,‘unknown’)
8.  *   contact: contact communication type (categorical: ‘cellular’,‘telephone’)
9.  *   month: last contact month of year (categorical: ‘jan’, ‘feb’, ‘mar’, …, ‘nov’, ‘dec’)
10.  *   day\_of\_week: last contact day of the week (categorical: ‘mon’,‘tue’,‘wed’,‘thu’,‘fri’)
11.  *   duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y=‘no’). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
12.  *   campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13.  *   pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14.  *   previous: number of contacts performed before this campaign and for this client (numeric)
15.  *   poutcome: outcome of the previous marketing campaign (categorical: ‘failure’,‘nonexistent’,‘success’)
16.  *   emp.var.rate: employment variation rate - quarterly indicator (numeric)
17.  *   cons.price.idx: consumer price index - monthly indicator (numeric)
18.  *   cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19.  *   euribor3m: euribor 3 month rate - daily indicator (numeric)
20.  *   nr.employed: number of employees - quarterly indicator (numeric)
21.  *   balance - average yearly balance, in euros (numeric)
22.  *   y - has the client subscribed a term deposit? (binary: ‘yes’,‘no’)

1.2 The SMART Questions
-----------------------

![](maxresdefault.jpg) The SMART questions are as follows:

1.Relationship between subscribing the term deposit and how much the customer is contacted (last contact, Campaign, Pdays, Previous Number of contacts)

2.  Find out the financially stable population? Will that affect the outcome?

3.Effect of dimensionality reduction on accuracy of the model.

4.  How are the likelihood of subscriptions affected by social and economic factors?

Throughout the paper we would try to answer the questions

Importing the required libraries

1.3 Importing the dataset
-------------------------

1.4 Basic Information about the data
------------------------------------

    Shape of dataset is : (45211, 23)
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 45211 entries, 0 to 45210
    Data columns (total 23 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   age             45211 non-null  int64  
     1   job             45211 non-null  object 
     2   marital         45211 non-null  object 
     3   education       45211 non-null  object 
     4   default         45211 non-null  object 
     5   balance         45211 non-null  int64  
     6   housing         45211 non-null  object 
     7   loan            45211 non-null  object 
     8   contact         45211 non-null  object 
     9   day             45211 non-null  int64  
     10  month           45211 non-null  object 
     11  duration        45211 non-null  int64  
     12  campaign        45211 non-null  int64  
     13  pdays           45211 non-null  int64  
     14  previous        45211 non-null  int64  
     15  poutcome        45211 non-null  object 
     16  y               45211 non-null  int64  
     17  month_int       45211 non-null  int64  
     18  cons.conf.idx   45211 non-null  float64
     19  emp.var.rate    45211 non-null  float64
     20  euribor3m       45211 non-null  float64
     21  nr.employed     45211 non-null  float64
     22  cons.price.idx  45211 non-null  float64
    dtypes: float64(5), int64(9), object(9)
    memory usage: 7.9+ MB
    Columns in dataset 
     None

2 Exploratory Data Analysis (EDA)
=================================

2.1 Distribution of y(target) variable
--------------------------------------

![](Summary_files/figure-html/cell-7-output-1.png)

We have 45,211 datapoints, if our model predicts only 0 as output, we would still get 88% accuracy, so our dataset is unbalanced which may gives misleading results. Along with the accuracy, we will also consider precision and recall for evaluation.

2.2 Missing values and Outliers
-------------------------------

### 2.2.1 Education

Here, even though we do not have any missing values but we have ‘unknown’ and ‘other’ as categories, so we will first get rid of them. The variables with ‘unknown’ rows are Education and Contact showned below.

    Text(0.5, 1.0, 'Type of education Distribution')

![](Summary_files/figure-html/cell-8-output-2.png)

### 2.2.2 Contact

    Text(0.5, 1.0, 'Type of Contact Distribution')

![](Summary_files/figure-html/cell-9-output-2.png)

*   since the type of communication(cellular and telephone) is not really a good indicator of subcription, we drop this variable.

### 2.2.3 Poutcome

![](Summary_files/figure-html/cell-10-output-1.png)

    poutcome
    failure     4901
    other       1840
    success     1511
    unknown    36959
    dtype: int64

There are _36959 unknown_ values(82%) and 1840 values with other(4.07% ) category, we will directly drop these columns.

2.3 Outliers
------------

![](Summary_files/figure-html/cell-11-output-1.png)

*   There are outliers in duration and balance so we need to get rid of them.

3 Data Cleaning
===============

*   Contact is not useful so we drop it.
*   In poutcome, we have a lot of ‘unknown’ and ‘other’ values so we drop it.  
    
*   Day is not giving any relevant infomation so we drop it.
*   Removing the unknowns from ‘job’ and ‘education’ columns.
*   Remove the outliers from balance and duration.

3.1 Dropping the irrelavant columns and missing values
------------------------------------------------------

    for job
    unknown : 288
    dropping rows with value as unknown in job
    for education
    unknown : 1730
    dropping rows with value as unknown in education

3.2 Outlier removal
-------------------

We have outliers in balance and duration, so to get rid of them we would try to remove the enteries few standard deviation away, since from the histograms most of the enteries are around mean only, we are removing the enteries more than 3SD away.

### 3.2.1 _Balance - Outliers_

    removing entries before balance   -7772.283533
    dtype: float64 and after balance    10480.338218
    dtype: float64

### 3.2.2 _Duration - Outliers_

Dropping rows where the duration of calls is less than 5sec since that is irrelevant. And also since converting the call duration in minutes rather than seconds makes more sense we would convert it into minutes.

plotting violen plot for duration and balance after cleaning data

![](Summary_files/figure-html/cell-15-output-1.png)

4 Data Visualization
====================

Let’ visualize important relationships between variables now.

4.1 SMART Question 1 :
----------------------

Relationship between subscribing the term deposit and how much the customer is contacted (last contact, Campaign, Pdays, Previous Number of contacts)

Answer : Based on last contact info only number of contacts performed during this campaign is contributing a lot towards subscription rates.

Suggestion: People who are contacted less than 5 times should be targeted more. Also, they could contact in less frequency in order to attract more target customers. The plot below shows the relationship between the number of calls and duration vs subscription

### 4.1.1 Number of calls versus Duration and affect on subscription

Here if we notice, people are more likely to subscribe if the number of calls are less than 5.

![](Summary_files/figure-html/cell-16-output-1.png)

Checking between pdays and previous as well

Here as we can see from the t- test, t

13.  *   pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14.  *   previous: number of contacts performed before this campaign and for this client (numeric)

We can notice from the plot that there is no relationship between subscription with pdays or previous. The datapoints are distrubuted randomly along the axies.

![](Summary_files/figure-html/cell-17-output-1.png)

4.2 Month wise subscription
---------------------------

    Text(0.5, 0, 'Month')

![](Summary_files/figure-html/cell-18-output-2.png)

Maximum percentage of people have subscribed in the month of March but bank is contacting people more in the month of May.

**Suggestion**:So it’s better to contact customer’s based on the subcription rate plot.

### 4.2.1 SMART Question 7: How are the likelihood of subscriptions affected by social and economic factors?

       month  cons.conf.idx  emp.var.rate  euribor3m  nr.employed
    0    jan           1310          1310       1310         1310
    1    feb           2492          2492       2492         2492
    2    mar            439           439        439          439
    3    apr           2772          2772       2772         2772
    4    may          13050         13050      13050        13050
    5    jun           4874          4874       4874         4874
    6    jul           6550          6550       6550         6550
    7    aug           5924          5924       5924         5924
    8    sep            514           514        514          514
    9    oct            661           661        661          661
    10   nov           3679          3679       3679         3679
    11   dec            195           195        195          195

**Answer** : Based on the above table we can see that there is no distinguishable difference in the month of march or may from rest of all the month, so social and economic factor **do not have major influence** on the outcome.

### 4.2.2 SMART Question 2

Find out the **financially stable** population? Will that affect the outcome?

We will try to find the financially stable population based on age, jobs, loan and balance.

### 4.2.3 Loan

    Text(0.5, 1.0, 'Type of loan Distribution')

![](Summary_files/figure-html/cell-21-output-2.png)

    Text(0.5, 1.0, 'Type of housing Distribution')

![](Summary_files/figure-html/cell-22-output-2.png)

People with housing loans are less likely to subscribe to term deposit but the difference here is not huge.

    Text(0.5, 1.0, 'Type of default Distribution')

![](Summary_files/figure-html/cell-23-output-2.png)

So people who have not paid back there loans and have credits, have not subcribed to the term deposit.

*   people who have loans are subscribing to term deposit less.

### 4.2.4 Age

Elder people might be more financially stable since they are subscriped to the term deposit more.

![](Summary_files/figure-html/cell-24-output-1.png)

*   People who are old are more likely to subscribe to term deposit.

### 4.2.5 Job

![](Summary_files/figure-html/cell-25-output-1.png)

![](Summary_files/figure-html/cell-25-output-2.png)

People in blue collar and management jobs are contacted more, which should not be the case. Since they have less subscription rates. Unlike popular assumption, students, retired and unemployment seem to have a high subscription rates. Even though they are contacted very less.

**suggestion**: The high subscripted rate group(students, retired and unemployment) should be contacted more.

### 4.2.6 Balance

Checking the subscriptions in each balance groups

               balGroup  % Contacted  % Subscription
    0       low balance    60.339143       10.503513
    1  moderate balance    17.399906       14.036275
    2      high balance    13.709374       16.715341
    3          Negative     8.551578        5.700909
           balanceGroup Contact Rate Subscription Rate
    0          Negative     8.551578          5.700909
    1       low balance    60.339143         10.503513
    2  moderate balance    17.399906         14.036275
    3      high balance    13.709374         16.715341

![](Summary_files/figure-html/cell-26-output-2.png)

**suggestion**:People with moderate to high balance, are contacted less but they have high subscription rates so bank should target them more.

It might be possible that balance group and jobs are telling the same information since some jobs might have high salary and thus balance groups might be depicting jobs only, so we will try to look at them together.

Balance Group versus Job

    Text(0.5, 1.0, 'Contact for each balance group in job category')

![](Summary_files/figure-html/cell-27-output-2.png)

![](Summary_files/figure-html/cell-27-output-3.png)

Student and Retired are more likely to subscribe and usually have moderate to high balance.

We found from the second bar chart that only the low balance groups are targeted in each category even though moderate to high balance category are more likely to subscribe.

5 Data Encoding
===============

5.1 One Hot Encoding
--------------------

We would encode ‘housing’,‘loan’,‘default’,‘job’,‘education’ and ‘marital’ as they are all categorical variables.

5.2 Sin - Cos encoding
----------------------

Transforming month into sin and cos so that there cyclic nature (jan-dec are as close as jan-feb) is retained which is usually lost in label encoding. Unlike one hot encoding, the dimension will reduce from 12(month\_jan, month\_feb … month\_dec) to 2(sin\_month , cos\_month)

    <AxesSubplot: xlabel='sin_month', ylabel='cos_month'>

![](Summary_files/figure-html/cell-30-output-2.png)

5.3 Dropping unnecessary columns irrelevant for modelling
---------------------------------------------------------

Here we dropped the ‘month’ column as they are encoded. Also, we dropped irrelvant variables ‘pdays’ and enconomic factors(‘cons.conf.idx’, ‘emp.var.rate’, ‘euribor3m’, ‘nr.employed’,‘cons.price.idx’) for modelling.

6 Data Modeling
===============

6.1 Splitting our Dataset
-------------------------

We are splitting our dataset in 1:4 ratio for training and testing set.

6.2 Balancing Our Dataset
-------------------------

We tried to balance our dataset using following methods:

*   Upsampling using SMOTE
*   Sin and cos transformation from month\_int.

7 Scaling numeric variables
===========================

Scaling age, balance, duration so that our algorithms perform better and all variables are treated equally. Since all three variables are in different scales, so we transform them into same standard.

8 Logistic Regression
=====================

Performing Logistic Regression on both balanced and unbalanced dataset. RFE is used in selecting the most important features ## Unbalanced Dataset

    Columns selected by RE ['duration', 'housing_no', 'housing_yes', 'loan_no', 'loan_yes', 'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_retired', 'job_services', 'job_student', 'education_primary', 'education_tertiary', 'cos_month', 'age', 'balance', 'sin_month']

As we can see from RFE, the most relevant features are :

*   Duration
*   Housing
*   Loan
*   Job
*   Education
*   cos\_month

From other features selection techniques and EDA, we can see that ‘age’ and ‘balance’ also contrubuted to the subscrption, so we added up these variables as well.

Applying model with selected features

    Accuracy for training set 0.8895725388601037
    Accuracy for testing set 0.8928403203014602
    Confusion matrix 
    [[7379  138]
     [ 772  203]]
                  precision    recall  f1-score   support
    
               0       0.91      0.98      0.94      7517
               1       0.60      0.21      0.31       975
    
        accuracy                           0.89      8492
       macro avg       0.75      0.59      0.63      8492
    weighted avg       0.87      0.89      0.87      8492
    

Here, the accuracy is 89% but the precision(0.59) and recall rate value(0.20) is low. And we also check on the balanced dataset since the low recall rate might be caused because of the less number of y = 1 value.

9 Balanced Dataset
==================

    Columns selected by RE ['housing_yes', 'loan_yes', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management', 'job_self-employed', 'job_services', 'job_technician', 'job_unemployed', 'education_primary', 'education_secondary', 'marital_divorced', 'marital_married', 'marital_single']

    Accuracy for training set 0.8824205094056934
    Accuracy for testing set 0.8220678285445124
    Confusion matrix 
    [[6356 1161]
     [ 350  625]]
                  precision    recall  f1-score   support
    
               0       0.95      0.85      0.89      7517
               1       0.35      0.64      0.45       975
    
        accuracy                           0.82      8492
       macro avg       0.65      0.74      0.67      8492
    weighted avg       0.88      0.82      0.84      8492
    

Here, important features are \* Housing \* Loan \* Job \* Education \* Marital Status

We also added the important features from unbalaced dataset \* Duration \* Age \* Month \* Balance

Here even though the precision and recall have improved, and accuracy has dropped down, but the important relationships are lost since the training data now is artificially generated datapoints. We will try to find the optimal cut-off value for original dataset and compare it with the model for balanced data.

### 9.0.1 Deciding cut off value for logistic regression - Unbalance

But to have good values for cut-off we would try to find a cutoff where the precision and recall values are decent

    
    Based on plot we would choose 0.25 as cut off 
    Accuracy for testing set 0.8777673104097975
    Confusion matrix 
    [[7018  499]
     [ 539  436]]
                  precision    recall  f1-score   support
    
               0       0.93      0.93      0.93      7517
               1       0.47      0.45      0.46       975
    
        accuracy                           0.88      8492
       macro avg       0.70      0.69      0.69      8492
    weighted avg       0.88      0.88      0.88      8492
    

![](Summary_files/figure-html/cell-41-output-2.png)

Optimal Cutoff at 0.25

Here as after applying feature selection, finding optimized cut-off, we are able to achieve higher accuracy with optimal precision and recall. Resulting from the comparison, we would continue our modellings with unbalance dataset.

### 9.0.2 Smart Question 5: The optimal cut off value for classification of our imbalance dataset.

**Answer**: The optimal cut off value for our imbalance dataset is 0.25 as the precision- recall chart indicated.

### 9.0.3 SMART Question 2: Since the dataset is imbalanced, will down sampling/up sampling or other techniques improve upon the accuracy of models.

**Answer**: As observed from above there is a slight improvement in accuracy, precision and recall after we apply SMOTE, but that improvement can also be acheived by adjusting the cut off value as well. So, we should always try adjusting cut-off first, before upsampling.

For ROC - AUC curve refer ([Figure 1](#fig-roc-curve)).  
For precision recall curve refer([Figure 2](#fig-pr-curve)).

10 Decision Tree
================

10.1 Feature Selection
----------------------

    Feature 0 variable age score 0.11
    Feature 1 variable balance score 0.15
    Feature 2 variable duration score 0.33
    Feature 3 variable campaign score 0.05
    Feature 4 variable previous score 0.04
    Feature 5 variable housing_no score 0.03
    Feature 6 variable housing_yes score 0.02
    Feature 11 variable job_admin. score 0.01
    Feature 15 variable job_management score 0.01
    Feature 20 variable job_technician score 0.01
    Feature 23 variable education_secondary score 0.01
    Feature 24 variable education_tertiary score 0.01
    Feature 25 variable marital_divorced score 0.01
    Feature 28 variable sin_month score 0.09
    Feature 29 variable cos_month score 0.03
    Important features from decision treee are : 
    ['age', 'balance', 'duration', 'campaign', 'previous', 'housing_no', 'housing_yes', 'job_admin.', 'job_management', 'job_technician', 'education_secondary', 'education_tertiary', 'marital_divorced', 'sin_month', 'cos_month']

![](Summary_files/figure-html/cell-42-output-2.png)

Features selected from this algorithm are

*   Age
*   Balance
*   Duration
*   Campaign
*   Previous
*   Housing
*   Job
*   Education
*   Marital
*   Month - Sin,cos

We have all the important features from EDA here

10.2 Hyperparameter tuning
--------------------------

For tuning the hyperparameter’s we will use GridSearch CV.

    Fitting 5 folds for each of 168 candidates, totalling 840 fits

    Best parameters from Grid Search CV : 
    {'criterion': 'entropy', 'max_depth': 6, 'max_features': 0.8, 'splitter': 'best'}

Training model based on the parameters we got from Grid SearchCV.

    0.8935468676401319
    [[7104  413]
     [ 491  484]]
                  precision    recall  f1-score   support
    
               0       0.94      0.95      0.94      7517
               1       0.54      0.50      0.52       975
    
        accuracy                           0.89      8492
       macro avg       0.74      0.72      0.73      8492
    weighted avg       0.89      0.89      0.89      8492
    

From the decision tree we have better precision, recall, accuracy and thus better f1 score. Hence, decision tree is performing better than logistic regression.

AUC Curve : [Figure 1](#fig-roc-curve)  
Precision Recall Curve : [Figure 2](#fig-pr-curve)

11 Random Forest
================

11.1 Feature Selection
----------------------

    Important features from random forest :
    ['age', 'balance', 'duration', 'campaign', 'previous', 'housing_no', 'housing_yes', 'job_admin.', 'job_management', 'job_technician', 'education_secondary', 'education_tertiary', 'marital_married', 'marital_single', 'sin_month', 'cos_month']

![](Summary_files/figure-html/cell-45-output-2.png)

11.2 Hyperparameter Tuning
--------------------------

    Fitting 3 folds for each of 32 candidates, totalling 96 fits

    {'bootstrap': True, 'max_depth': 90, 'max_features': 3, 'n_estimators': 300}

    Training accuracy 1.0
    Testing set accuracy 0.8986104569006124
    [[7273  244]
     [ 617  358]]
                  precision    recall  f1-score   support
    
               0       0.92      0.97      0.94      7517
               1       0.59      0.37      0.45       975
    
        accuracy                           0.90      8492
       macro avg       0.76      0.67      0.70      8492
    weighted avg       0.88      0.90      0.89      8492
    

We are getting best performance from Random Forest but we are not sure why we are getting such idealistic results so we would also apply cross validation to test our results

    {'Training Accuracy scores': array([1., 1., 1., 1., 1.]),
     'Mean Training Accuracy': 100.0,
     'Training Precision scores': array([1., 1., 1., 1., 1.]),
     'Mean Training Precision': 1.0,
     'Training Recall scores': array([1., 1., 1., 1., 1.]),
     'Mean Training Recall': 1.0,
     'Training F1 scores': array([1., 1., 1., 1., 1.]),
     'Mean Training F1 Score': 1.0,
     'Validation Accuracy scores': array([0.89549603, 0.90035325, 0.89696791, 0.89842485, 0.89842485]),
     'Mean Validation Accuracy': 89.79333779716873,
     'Validation Precision scores': array([0.58850575, 0.61956522, 0.58969072, 0.59958506, 0.61009174]),
     'Mean Validation Precision': 0.6014876983054311,
     'Validation Recall scores': array([0.3252859 , 0.36213469, 0.36340534, 0.36768448, 0.33842239]),
     'Mean Validation Recall': 0.35138655828976595,
     'Validation F1 scores': array([0.41898527, 0.45709703, 0.44968553, 0.45583596, 0.43535188]),
     'Mean Validation F1 Score': 0.4433911363649415}

After applying cross validation, we are getting some what real estimates.

AUC Curve : [Figure 1](#fig-roc-curve)  
Precision Recall Curve : [Figure 2](#fig-pr-curve)

12 Linear SVC
=============

Finding a linear hyperplane that tries to separate two classes.

    0.8906029203956665
    [[7413  104]
     [ 825  150]]
                  precision    recall  f1-score   support
    
               0       0.90      0.99      0.94      7517
               1       0.59      0.15      0.24       975
    
        accuracy                           0.89      8492
       macro avg       0.75      0.57      0.59      8492
    weighted avg       0.86      0.89      0.86      8492
    

13 SVC
======

Finding a complex hyperplane that tries to separate the classes.

    0.8922515308525671
    [[7446   71]
     [ 844  131]]
                  precision    recall  f1-score   support
    
               0       0.90      0.99      0.94      7517
               1       0.65      0.13      0.22       975
    
        accuracy                           0.89      8492
       macro avg       0.77      0.56      0.58      8492
    weighted avg       0.87      0.89      0.86      8492
    

14 Naive Bayes
==============

Naive Bayes a naive assumption that all the features are independent of each other and thus by reducing the complexity of computing conditional probabilities it evaluates the probability of 0 and 1.

    Fitting 10 folds for each of 100 candidates, totalling 1000 fits

    GaussianNB(var_smoothing=0.0533669923120631)
    Model score is 0.8871879415920867

![](Summary_files/figure-html/cell-51-output-3.png)

    test set evaluation: 
    0.8871879415920867
    [[7291  226]
     [ 732  243]]
                  precision    recall  f1-score   support
    
               0       0.91      0.97      0.94      7517
               1       0.52      0.25      0.34       975
    
        accuracy                           0.89      8492
       macro avg       0.71      0.61      0.64      8492
    weighted avg       0.86      0.89      0.87      8492
    

14.1 For balanced
-----------------

For balanced dataset, as we can see there is a slight improvement in performance. The f1 score has improved and also, the yellow bars are now slightly shifted towards right side.

    Model score is 0.459609043805935

![](Summary_files/figure-html/cell-52-output-2.png)

    test set evaluation: 
    0.459609043805935
    [[2994 4523]
     [  66  909]]
                  precision    recall  f1-score   support
    
               0       0.98      0.40      0.57      7517
               1       0.17      0.93      0.28       975
    
        accuracy                           0.46      8492
       macro avg       0.57      0.67      0.42      8492
    weighted avg       0.89      0.46      0.53      8492
    

As we can see from the graph for the red and yellow bars for yes(1 term deposit) are coming on the opposite sides which is not expected.

AUC Curve : [Figure 1](#fig-roc-curve)  
Precision Recall Curve : [Figure 2](#fig-pr-curve)

15 KNN
======

Using the k - nearest neighbours we try to predict the testing dataset. Now to find the optimal k value we will look into precision and accuracy curve for different k values.

    Maximum accuracy:- 0.8936646255299106 at K = 16

![](Summary_files/figure-html/cell-54-output-2.png)

Accuracy curve for different k values

    Maximum Precision:- 0.23186915390816643 at K = 6

![](Summary_files/figure-html/cell-55-output-2.png)

Precision curve for different k values

Based on the above plot, optimal k value is 3, with maximum f1 score of 0.64.

    Train set accuracy 0.9290508714083844
    Test set accuracy 0.8795336787564767
    [[7202  315]
     [ 708  267]]
                  precision    recall  f1-score   support
    
               0       0.91      0.96      0.93      7517
               1       0.46      0.27      0.34       975
    
        accuracy                           0.88      8492
       macro avg       0.68      0.62      0.64      8492
    weighted avg       0.86      0.88      0.87      8492
    

AUC Curve : [Figure 1](#fig-roc-curve)  
Precision Recall Curve : [Figure 2](#fig-pr-curve)

16 ROC -AUC Curve
=================

![](Summary_files/figure-html/fig-roc-curve-output-1.png)

Figure 1: AUC ROC Curve for all Models

17 Precision Recall Curve
=========================

In imbalance problem since we have a high number of Negatives, this makes the False Posiitve Rate as low, resulting in the shift of ROC AUC Curve towards left, which is slightly misleading.

So in imbalance problem we usually make sure to look at precision recall curve as well.

![](Summary_files/figure-html/fig-pr-curve-output-1.png)

Figure 2: Precision Recall Curve for all Models

As per the ROC Curve and Precision Recall curve, KNN is performing best. But after combining these results with precision recall curve, we suggest using Random Forest for our problem.

18 Summary
==========

Table 1: Summary of Models

Model

Accuracy

Precision

Recall

AUC

Logistic(Cutoff=0.25)

0.88

0.51

0.58

0.872

Logistic (Balanced-Train)

0.85

0.49

0.54

Decision Tree

0.91

0.66

0.47

0.923

Random Forest

0.88

0.66

0.46

0.913

SVC

0.89

0.75

0.15

Linear SVC

0.89

0.62

0.16

Gaussian Bayes

0.88

0.50

0.25

0.841

KNN

0.92

0.78

0.54

0.965

Naive Bayes

0.85

0.56

0.02

Naive Bayes (Balanced-Train)

0.69

0.19

0.35

See [Table 1](#tbl-letters).

19 Conclusion
=============

Our model would be beneficial in the following ways :

*   For target marketing for bank campaigns, or in other events. For example based on the customer’s job, age and loan history the model would can easily predict whether the customer is going to subscribe to the term deposit or not. So out of the million people, we can easily shortlist people based on our model and spend the time on them so as to improve efficiency.
    
*   Improving buissness effficiency of banks. Since using the eda or model we can easily check the subscription insights, it would be very helpful for banks to improve their stratergies. For example, based on the monthly subscription rates, if banks are deciding the campaign promotion time, it can improve there efficiency.
    
*   Since, we have month as a input factor in our model, and all other values are static, we can even find the best month to contact customer based on the predicted probability of the customer. As there can be a relation between the job type and the month they are subscribing or their fluctuating balance and age. This can be very useful in finding the best time to contact.
    
*   Based on the model, since the number of contact is playing a major role, if we have the optimal time to contact them, we can restrict our calls to less than 5 and find a better turnover.
    
*   We didn’t see any relation with the social and economic factors here, but if we had the data for multiple years, there was a possibility of finding a relation. Our model can accomodate these factors as well, and if trained by accomodating these factors as well, this can be helpful for banks in finding the proper time for there campaign.
    

Hence, analyzing this kind of marketing dataset has given us valuable insight into how we can tweak our model to give buisness insights as well as customer insights to improve subscription of term deposits.

20 Reference
============

*   https://www.kaggle.com/janiobachmann/bank-marketing-dataset
    
*   (PDF) Data Analysis of a Portuguese marketing campaign using bank … (no date). Available at: https://www.researchgate.net/publication/339988208\_Data\_Analysis\_of\_a\_Portuguese\_Marketing\_Campaign\_using\_Bank\_Marketing\_data\_Set (Accessed: December 20, 2022).
    
*   Bank marketing data set. (n.d.). 1010data.com. Retrieved December 20, 2022, from https://docs.1010data.com/Tutorials/MachineLearningExamples/BankMarketingDataSet\_2.html
    
*   Manda, H., Srinivasan, S., & Rangarao, D. (2021). IBM Cloud Pak for Data: An enterprise platform to operationalize data, analytics, and AI. Packt Publishing.
    
*   Solving Bank Marketing Calssification Problem - Databricks. (n.d.). Databricks.com. Retrieved December 20, 2022, from https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/8143187682226564/2297613386094950/3186001515933643/latest.html
    
*   Solving Bank Marketing Calssification Problem - Databricks. (n.d.). Databricks.com. Retrieved December 20, 2022, from https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/8143187682226564/2297613386094950/3186001515933643/latest.html
    
*   Bank Marketing Data Set. (n.d.). UCI Machine Learning Repository. Retrieved December 20, 2022, from https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
    
*   https://tradingeconomics.com/
    

window.document.addEventListener("DOMContentLoaded", function (event) { const toggleBodyColorMode = (bsSheetEl) => { const mode = bsSheetEl.getAttribute("data-mode"); const bodyEl = window.document.querySelector("body"); if (mode === "dark") { bodyEl.classList.add("quarto-dark"); bodyEl.classList.remove("quarto-light"); } else { bodyEl.classList.add("quarto-light"); bodyEl.classList.remove("quarto-dark"); } } const toggleBodyColorPrimary = () => { const bsSheetEl = window.document.querySelector("link#quarto-bootstrap"); if (bsSheetEl) { toggleBodyColorMode(bsSheetEl); } } toggleBodyColorPrimary(); const icon = ""; const anchorJS = new window.AnchorJS(); anchorJS.options = { placement: 'right', icon: icon }; anchorJS.add('.anchored'); const clipboard = new window.ClipboardJS('.code-copy-button', { target: function(trigger) { return trigger.previousElementSibling; } }); clipboard.on('success', function(e) { // button target const button = e.trigger; // don't keep focus button.blur(); // flash "checked" button.classList.add('code-copy-button-checked'); var currentTitle = button.getAttribute("title"); button.setAttribute("title", "Copied!"); setTimeout(function() { button.setAttribute("title", currentTitle); button.classList.remove('code-copy-button-checked'); }, 1000); // clear code selection e.clearSelection(); }); function tippyHover(el, contentFn) { const config = { allowHTML: true, content: contentFn, maxWidth: 500, delay: 100, arrow: false, appendTo: function(el) { return el.parentElement; }, interactive: true, interactiveBorder: 10, theme: 'quarto', placement: 'bottom-start' }; window.tippy(el, config); } const noterefs = window.document.querySelectorAll('a\[role="doc-noteref"\]'); for (var i=0; i<noterefs.length; i++) { const ref = noterefs\[i\]; tippyHover(ref, function() { // use id or data attribute instead here let href = ref.getAttribute('data-footnote-href') || ref.getAttribute('href'); try { href = new URL(href).hash; } catch {} const id = href.replace(/^#\\/?/, ""); const note = window.document.getElementById(id); return note.innerHTML;<!DOCTYPE html> <html xmlns="http://www.w3.org/1999/xhtml" lang="en" xml:lang="en"><head> <meta charset="utf-8"> <meta name="generator" content="quarto-1.1.251"> <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes"> <meta name="author" content="TEAM 11: Anjali Mudgal, Guoshan Yu, Medhasweta Sen"> <meta name="dcterms.date" content="2022-12-20"> <title>AN ANALYSIS OF PORTUGUESE BANK MARKETING DATA</title> <style> code{white-space: pre-wrap;} span.smallcaps{font-variant: small-caps;} div.columns{display: flex; gap: min(4vw, 1.5em);} div.column{flex: auto; overflow-x: auto;} div.hanging-indent{margin-left: 1.5em; text-indent: -1.5em;} ul.task-list{list-style: none;} ul.task-list li input\[type="checkbox"\] { width: 0.8em; margin: 0 0.8em 0.2em -1.6em; vertical-align: middle; } </style> <script src="Summary\_files/libs/clipboard/clipboard.min.js">     html{ scroll-behavior: smooth; }

Table of contents
-----------------

*   [1 INTRODUCTION](#introduction)
    *   [1.1 The Data Set](#the-data-set)
    *   [1.2 The SMART Questions](#the-smart-questions)
    *   [1.3 Importing the dataset](#importing-the-dataset)
    *   [1.4 Basic Information about the data](#basic-information-about-the-data)
*   [2 Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    *   [2.1 Distribution of y(target) variable](#distribution-of-ytarget-variable)
    *   [2.2 Missing values and Outliers](#missing-values-and-outliers)
        *   [2.2.1 Education](#education)
        *   [2.2.2 Contact](#contact)
        *   [2.2.3 Poutcome](#poutcome)
    *   [2.3 Outliers](#outliers)
*   [3 Data Cleaning](#data-cleaning)
    *   [3.1 Dropping the irrelavant columns and missing values](#dropping-the-irrelavant-columns-and-missing-values)
    *   [3.2 Outlier removal](#outlier-removal)
        *   [3.2.1 _Balance - Outliers_](#balance---outliers)
        *   [3.2.2 _Duration - Outliers_](#duration---outliers)
*   [4 Data Visualization](#data-visualization)
    *   [4.1 SMART Question 1 :](#smart-question-1)
        *   [4.1.1 Number of calls versus Duration and affect on subscription](#number-of-calls-versus-duration-and-affect-on-subscription)
    *   [4.2 Month wise subscription](#month-wise-subscription)
        *   [4.2.1 SMART Question 7: How are the likelihood of subscriptions affected by social and economic factors?](#smart-question-7-how-are-the-likelihood-of-subscriptions-affected-by-social-and-economic-factors)
        *   [4.2.2 SMART Question 2](#smart-question-2)
        *   [4.2.3 Loan](#loan)
        *   [4.2.4 Age](#age)
        *   [4.2.5 Job](#job)
        *   [4.2.6 Balance](#balance)
*   [5 Data Encoding](#data-encoding)
    *   [5.1 One Hot Encoding](#one-hot-encoding)
    *   [5.2 Sin - Cos encoding](#sin---cos-encoding)
    *   [5.3 Dropping unnecessary columns irrelevant for modelling](#dropping-unnecessary-columns-irrelevant-for-modelling)
*   [6 Data Modeling](#data-modeling)
    *   [6.1 Splitting our Dataset](#splitting-our-dataset)
    *   [6.2 Balancing Our Dataset](#balancing-our-dataset)
*   [7 Scaling numeric variables](#scaling-numeric-variables)
*   [8 Logistic Regression](#logistic-regression)
*   [9 Balanced Dataset](#balanced-dataset)
    *   [9.0.1 Deciding cut off value for logistic regression - Unbalance](#deciding-cut-off-value-for-logistic-regression---unbalance)
    *   [9.0.2 Smart Question 5: The optimal cut off value for classification of our imbalance dataset.](#smart-question-5-the-optimal-cut-off-value-for-classification-of-our-imbalance-dataset.)
    *   [9.0.3 SMART Question 2: Since the dataset is imbalanced, will down sampling/up sampling or other techniques improve upon the accuracy of models.](#smart-question-2-since-the-dataset-is-imbalanced-will-down-samplingup-sampling-or-other-techniques-improve-upon-the-accuracy-of-models.)
*   [10 Decision Tree](#decision-tree)
    *   [10.1 Feature Selection](#feature-selection)
    *   [10.2 Hyperparameter tuning](#hyperparameter-tuning)
*   [11 Random Forest](#random-forest)
    *   [11.1 Feature Selection](#feature-selection-1)
    *   [11.2 Hyperparameter Tuning](#hyperparameter-tuning-1)
*   [12 Linear SVC](#linear-svc)
*   [13 SVC](#svc)
*   [14 Naive Bayes](#naive-bayes)
    *   [14.1 For balanced](#for-balanced)
*   [15 KNN](#knn)
*   [16 ROC -AUC Curve](#roc--auc-curve)
*   [17 Precision Recall Curve](#precision-recall-curve)
*   [18 Summary](#summary)
*   [19 Conclusion](#conclusion)
*   [20 Reference](#reference)

AN ANALYSIS OF PORTUGUESE BANK MARKETING DATA
=============================================

The George Washington University (DATS 6103: An Introduction to Data Mining)

Author

TEAM 11: Anjali Mudgal, Guoshan Yu, Medhasweta Sen

Published

December 20, 2022

1 INTRODUCTION
==============

Bank marketing is the practice of attracting and acquiring new customers through traditional media and digital media strategies. The use of these media strategies helps determine what kind of customer is attracted to a certain institutions. This also includes different banking institutions purposefully using different strategies to attract the type of customer they want to do business with.

Marketing has evolved from a communication role to a revenue generating role. The consumer has evolved from being a passive recipient of marketing messages to an active participant in the marketing process. Technology has evolved from being a means of communication to a means of data collection and analysis. Data analytics has evolved from being a means of understanding the consumer to a means of understanding the consumer and the institution.

Bank marketing strategy is increasingly focused on digital channels, including social media, video, search and connected TV. As bank and credit union marketers strive to promote brand awareness, they need a new way to assess channel ROI and more accurate data to enable personalized offers. Add to that the growing importance of purpose-driven marketing.

The relentless pace of digitization is disrupting not only the established order in banking, but bank marketing strategies. Marketers at both traditional institutions and digital disruptors are feeling the pressure.

Just as bank marketers begin to master one channel, consumers move to another. Many now toggle between devices on a seemingly infinite number of platforms, making it harder than ever for marketers to pin down the right consumers at the right time in the right place.

![](expected-marketing-budget-changes-by-channel.png)

1.1 The Data Set
----------------

The data set used in this analysis is from a Portuguese bank. The data set contains 41,188 observations and 21 variables. The variables include the following:

1.  *   age (numeric)
2.  *   job : type of job (categorical: ‘admin.’,‘blue-collar’,‘entrepreneur’,‘housemaid’,‘management’,‘retired’,‘self-employed’,‘services’,‘student’,‘technician’,‘unemployed’,‘unknown’)
3.  *   marital : marital status (categorical: ‘divorced’,‘married’,‘single’,‘unknown’; note: ‘divorced’ means divorced or widowed)
4.  *   education (categorical: ‘basic.4y’,‘basic.6y’,‘basic.9y’,‘high.school’,‘illiterate’,‘professional.course’,‘university.degree’,‘unknown’)
5.  *   default: has credit in default? (categorical: ‘no’,‘yes’,‘unknown’)
6.  *   housing: has housing loan? (categorical: ‘no’,‘yes’,‘unknown’)
7.  *   loan: has personal loan? (categorical: ‘no’,‘yes’,‘unknown’)
8.  *   contact: contact communication type (categorical: ‘cellular’,‘telephone’)
9.  *   month: last contact month of year (categorical: ‘jan’, ‘feb’, ‘mar’, …, ‘nov’, ‘dec’)
10.  *   day\_of\_week: last contact day of the week (categorical: ‘mon’,‘tue’,‘wed’,‘thu’,‘fri’)
11.  *   duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y=‘no’). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
12.  *   campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13.  *   pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14.  *   previous: number of contacts performed before this campaign and for this client (numeric)
15.  *   poutcome: outcome of the previous marketing campaign (categorical: ‘failure’,‘nonexistent’,‘success’)
16.  *   emp.var.rate: employment variation rate - quarterly indicator (numeric)
17.  *   cons.price.idx: consumer price index - monthly indicator (numeric)
18.  *   cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19.  *   euribor3m: euribor 3 month rate - daily indicator (numeric)
20.  *   nr.employed: number of employees - quarterly indicator (numeric)
21.  *   balance - average yearly balance, in euros (numeric)
22.  *   y - has the client subscribed a term deposit? (binary: ‘yes’,‘no’)

1.2 The SMART Questions
-----------------------

![](maxresdefault.jpg) The SMART questions are as follows:

1.Relationship between subscribing the term deposit and how much the customer is contacted (last contact, Campaign, Pdays, Previous Number of contacts)

2.  Find out the financially stable population? Will that affect the outcome?

3.Effect of dimensionality reduction on accuracy of the model.

4.  How are the likelihood of subscriptions affected by social and economic factors?

Throughout the paper we would try to answer the questions

Importing the required libraries

1.3 Importing the dataset
-------------------------

1.4 Basic Information about the data
------------------------------------

    Shape of dataset is : (45211, 23)
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 45211 entries, 0 to 45210
    Data columns (total 23 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   age             45211 non-null  int64  
     1   job             45211 non-null  object 
     2   marital         45211 non-null  object 
     3   education       45211 non-null  object 
     4   default         45211 non-null  object 
     5   balance         45211 non-null  int64  
     6   housing         45211 non-null  object 
     7   loan            45211 non-null  object 
     8   contact         45211 non-null  object 
     9   day             45211 non-null  int64  
     10  month           45211 non-null  object 
     11  duration        45211 non-null  int64  
     12  campaign        45211 non-null  int64  
     13  pdays           45211 non-null  int64  
     14  previous        45211 non-null  int64  
     15  poutcome        45211 non-null  object 
     16  y               45211 non-null  int64  
     17  month_int       45211 non-null  int64  
     18  cons.conf.idx   45211 non-null  float64
     19  emp.var.rate    45211 non-null  float64
     20  euribor3m       45211 non-null  float64
     21  nr.employed     45211 non-null  float64
     22  cons.price.idx  45211 non-null  float64
    dtypes: float64(5), int64(9), object(9)
    memory usage: 7.9+ MB
    Columns in dataset 
     None

2 Exploratory Data Analysis (EDA)
=================================

2.1 Distribution of y(target) variable
--------------------------------------

![](Summary_files/figure-html/cell-7-output-1.png)

We have 45,211 datapoints, if our model predicts only 0 as output, we would still get 88% accuracy, so our dataset is unbalanced which may gives misleading results. Along with the accuracy, we will also consider precision and recall for evaluation.

2.2 Missing values and Outliers
-------------------------------

### 2.2.1 Education

Here, even though we do not have any missing values but we have ‘unknown’ and ‘other’ as categories, so we will first get rid of them. The variables with ‘unknown’ rows are Education and Contact showned below.

    Text(0.5, 1.0, 'Type of education Distribution')

![](Summary_files/figure-html/cell-8-output-2.png)

### 2.2.2 Contact

    Text(0.5, 1.0, 'Type of Contact Distribution')

![](Summary_files/figure-html/cell-9-output-2.png)

*   since the type of communication(cellular and telephone) is not really a good indicator of subcription, we drop this variable.

### 2.2.3 Poutcome

![](Summary_files/figure-html/cell-10-output-1.png)

    poutcome
    failure     4901
    other       1840
    success     1511
    unknown    36959
    dtype: int64

There are _36959 unknown_ values(82%) and 1840 values with other(4.07% ) category, we will directly drop these columns.

2.3 Outliers
------------

![](Summary_files/figure-html/cell-11-output-1.png)

*   There are outliers in duration and balance so we need to get rid of them.

3 Data Cleaning
===============

*   Contact is not useful so we drop it.
*   In poutcome, we have a lot of ‘unknown’ and ‘other’ values so we drop it.  
    
*   Day is not giving any relevant infomation so we drop it.
*   Removing the unknowns from ‘job’ and ‘education’ columns.
*   Remove the outliers from balance and duration.

3.1 Dropping the irrelavant columns and missing values
------------------------------------------------------

    for job
    unknown : 288
    dropping rows with value as unknown in job
    for education
    unknown : 1730
    dropping rows with value as unknown in education

3.2 Outlier removal
-------------------

We have outliers in balance and duration, so to get rid of them we would try to remove the enteries few standard deviation away, since from the histograms most of the enteries are around mean only, we are removing the enteries more than 3SD away.

### 3.2.1 _Balance - Outliers_

    removing entries before balance   -7772.283533
    dtype: float64 and after balance    10480.338218
    dtype: float64

### 3.2.2 _Duration - Outliers_

Dropping rows where the duration of calls is less than 5sec since that is irrelevant. And also since converting the call duration in minutes rather than seconds makes more sense we would convert it into minutes.

plotting violen plot for duration and balance after cleaning data

![](Summary_files/figure-html/cell-15-output-1.png)

4 Data Visualization
====================

Let’ visualize important relationships between variables now.

4.1 SMART Question 1 :
----------------------

Relationship between subscribing the term deposit and how much the customer is contacted (last contact, Campaign, Pdays, Previous Number of contacts)

Answer : Based on last contact info only number of contacts performed during this campaign is contributing a lot towards subscription rates.

Suggestion: People who are contacted less than 5 times should be targeted more. Also, they could contact in less frequency in order to attract more target customers. The plot below shows the relationship between the number of calls and duration vs subscription

### 4.1.1 Number of calls versus Duration and affect on subscription

Here if we notice, people are more likely to subscribe if the number of calls are less than 5.

![](Summary_files/figure-html/cell-16-output-1.png)

Checking between pdays and previous as well

Here as we can see from the t- test, t

13.  *   pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14.  *   previous: number of contacts performed before this campaign and for this client (numeric)

We can notice from the plot that there is no relationship between subscription with pdays or previous. The datapoints are distrubuted randomly along the axies.

![](Summary_files/figure-html/cell-17-output-1.png)

4.2 Month wise subscription
---------------------------

    Text(0.5, 0, 'Month')

![](Summary_files/figure-html/cell-18-output-2.png)

Maximum percentage of people have subscribed in the month of March but bank is contacting people more in the month of May.

**Suggestion**:So it’s better to contact customer’s based on the subcription rate plot.

### 4.2.1 SMART Question 7: How are the likelihood of subscriptions affected by social and economic factors?

       month  cons.conf.idx  emp.var.rate  euribor3m  nr.employed
    0    jan           1310          1310       1310         1310
    1    feb           2492          2492       2492         2492
    2    mar            439           439        439          439
    3    apr           2772          2772       2772         2772
    4    may          13050         13050      13050        13050
    5    jun           4874          4874       4874         4874
    6    jul           6550          6550       6550         6550
    7    aug           5924          5924       5924         5924
    8    sep            514           514        514          514
    9    oct            661           661        661          661
    10   nov           3679          3679       3679         3679
    11   dec            195           195        195          195

**Answer** : Based on the above table we can see that there is no distinguishable difference in the month of march or may from rest of all the month, so social and economic factor **do not have major influence** on the outcome.

### 4.2.2 SMART Question 2

Find out the **financially stable** population? Will that affect the outcome?

We will try to find the financially stable population based on age, jobs, loan and balance.

### 4.2.3 Loan

    Text(0.5, 1.0, 'Type of loan Distribution')

![](Summary_files/figure-html/cell-21-output-2.png)

    Text(0.5, 1.0, 'Type of housing Distribution')

![](Summary_files/figure-html/cell-22-output-2.png)

People with housing loans are less likely to subscribe to term deposit but the difference here is not huge.

    Text(0.5, 1.0, 'Type of default Distribution')

![](Summary_files/figure-html/cell-23-output-2.png)

So people who have not paid back there loans and have credits, have not subcribed to the term deposit.

*   people who have loans are subscribing to term deposit less.

### 4.2.4 Age

Elder people might be more financially stable since they are subscriped to the term deposit more.

![](Summary_files/figure-html/cell-24-output-1.png)

*   People who are old are more likely to subscribe to term deposit.

### 4.2.5 Job

![](Summary_files/figure-html/cell-25-output-1.png)

![](Summary_files/figure-html/cell-25-output-2.png)

People in blue collar and management jobs are contacted more, which should not be the case. Since they have less subscription rates. Unlike popular assumption, students, retired and unemployment seem to have a high subscription rates. Even though they are contacted very less.

**suggestion**: The high subscripted rate group(students, retired and unemployment) should be contacted more.

### 4.2.6 Balance

Checking the subscriptions in each balance groups

               balGroup  % Contacted  % Subscription
    0       low balance    60.339143       10.503513
    1  moderate balance    17.399906       14.036275
    2      high balance    13.709374       16.715341
    3          Negative     8.551578        5.700909
           balanceGroup Contact Rate Subscription Rate
    0          Negative     8.551578          5.700909
    1       low balance    60.339143         10.503513
    2  moderate balance    17.399906         14.036275
    3      high balance    13.709374         16.715341

![](Summary_files/figure-html/cell-26-output-2.png)

**suggestion**:People with moderate to high balance, are contacted less but they have high subscription rates so bank should target them more.

It might be possible that balance group and jobs are telling the same information since some jobs might have high salary and thus balance groups might be depicting jobs only, so we will try to look at them together.

Balance Group versus Job

    Text(0.5, 1.0, 'Contact for each balance group in job category')

![](Summary_files/figure-html/cell-27-output-2.png)

![](Summary_files/figure-html/cell-27-output-3.png)

Student and Retired are more likely to subscribe and usually have moderate to high balance.

We found from the second bar chart that only the low balance groups are targeted in each category even though moderate to high balance category are more likely to subscribe.

5 Data Encoding
===============

5.1 One Hot Encoding
--------------------

We would encode ‘housing’,‘loan’,‘default’,‘job’,‘education’ and ‘marital’ as they are all categorical variables.

5.2 Sin - Cos encoding
----------------------

Transforming month into sin and cos so that there cyclic nature (jan-dec are as close as jan-feb) is retained which is usually lost in label encoding. Unlike one hot encoding, the dimension will reduce from 12(month\_jan, month\_feb … month\_dec) to 2(sin\_month , cos\_month)

    <AxesSubplot: xlabel='sin_month', ylabel='cos_month'>

![](Summary_files/figure-html/cell-30-output-2.png)

5.3 Dropping unnecessary columns irrelevant for modelling
---------------------------------------------------------

Here we dropped the ‘month’ column as they are encoded. Also, we dropped irrelvant variables ‘pdays’ and enconomic factors(‘cons.conf.idx’, ‘emp.var.rate’, ‘euribor3m’, ‘nr.employed’,‘cons.price.idx’) for modelling.

6 Data Modeling
===============

6.1 Splitting our Dataset
-------------------------

We are splitting our dataset in 1:4 ratio for training and testing set.

6.2 Balancing Our Dataset
-------------------------

We tried to balance our dataset using following methods:

*   Upsampling using SMOTE
*   Sin and cos transformation from month\_int.

7 Scaling numeric variables
===========================

Scaling age, balance, duration so that our algorithms perform better and all variables are treated equally. Since all three variables are in different scales, so we transform them into same standard.

8 Logistic Regression
=====================

Performing Logistic Regression on both balanced and unbalanced dataset. RFE is used in selecting the most important features ## Unbalanced Dataset

    Columns selected by RE ['duration', 'housing_no', 'housing_yes', 'loan_no', 'loan_yes', 'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_retired', 'job_services', 'job_student', 'education_primary', 'education_tertiary', 'cos_month', 'age', 'balance', 'sin_month']

As we can see from RFE, the most relevant features are :

*   Duration
*   Housing
*   Loan
*   Job
*   Education
*   cos\_month

From other features selection techniques and EDA, we can see that ‘age’ and ‘balance’ also contrubuted to the subscrption, so we added up these variables as well.

Applying model with selected features

    Accuracy for training set 0.8895725388601037
    Accuracy for testing set 0.8928403203014602
    Confusion matrix 
    [[7379  138]
     [ 772  203]]
                  precision    recall  f1-score   support
    
               0       0.91      0.98      0.94      7517
               1       0.60      0.21      0.31       975
    
        accuracy                           0.89      8492
       macro avg       0.75      0.59      0.63      8492
    weighted avg       0.87      0.89      0.87      8492
    

Here, the accuracy is 89% but the precision(0.59) and recall rate value(0.20) is low. And we also check on the balanced dataset since the low recall rate might be caused because of the less number of y = 1 value.

9 Balanced Dataset
==================

    Columns selected by RE ['housing_yes', 'loan_yes', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management', 'job_self-employed', 'job_services', 'job_technician', 'job_unemployed', 'education_primary', 'education_secondary', 'marital_divorced', 'marital_married', 'marital_single']

    Accuracy for training set 0.8824205094056934
    Accuracy for testing set 0.8220678285445124
    Confusion matrix 
    [[6356 1161]
     [ 350  625]]
                  precision    recall  f1-score   support
    
               0       0.95      0.85      0.89      7517
               1       0.35      0.64      0.45       975
    
        accuracy                           0.82      8492
       macro avg       0.65      0.74      0.67      8492
    weighted avg       0.88      0.82      0.84      8492
    

Here, important features are \* Housing \* Loan \* Job \* Education \* Marital Status

We also added the important features from unbalaced dataset \* Duration \* Age \* Month \* Balance

Here even though the precision and recall have improved, and accuracy has dropped down, but the important relationships are lost since the training data now is artificially generated datapoints. We will try to find the optimal cut-off value for original dataset and compare it with the model for balanced data.

### 9.0.1 Deciding cut off value for logistic regression - Unbalance

But to have good values for cut-off we would try to find a cutoff where the precision and recall values are decent

    
    Based on plot we would choose 0.25 as cut off 
    Accuracy for testing set 0.8777673104097975
    Confusion matrix 
    [[7018  499]
     [ 539  436]]
                  precision    recall  f1-score   support
    
               0       0.93      0.93      0.93      7517
               1       0.47      0.45      0.46       975
    
        accuracy                           0.88      8492
       macro avg       0.70      0.69      0.69      8492
    weighted avg       0.88      0.88      0.88      8492
    

![](Summary_files/figure-html/cell-41-output-2.png)

Optimal Cutoff at 0.25

Here as after applying feature selection, finding optimized cut-off, we are able to achieve higher accuracy with optimal precision and recall. Resulting from the comparison, we would continue our modellings with unbalance dataset.

### 9.0.2 Smart Question 5: The optimal cut off value for classification of our imbalance dataset.

**Answer**: The optimal cut off value for our imbalance dataset is 0.25 as the precision- recall chart indicated.

### 9.0.3 SMART Question 2: Since the dataset is imbalanced, will down sampling/up sampling or other techniques improve upon the accuracy of models.

**Answer**: As observed from above there is a slight improvement in accuracy, precision and recall after we apply SMOTE, but that improvement can also be acheived by adjusting the cut off value as well. So, we should always try adjusting cut-off first, before upsampling.

For ROC - AUC curve refer ([Figure 1](#fig-roc-curve)).  
For precision recall curve refer([Figure 2](#fig-pr-curve)).

10 Decision Tree
================

10.1 Feature Selection
----------------------

    Feature 0 variable age score 0.11
    Feature 1 variable balance score 0.15
    Feature 2 variable duration score 0.33
    Feature 3 variable campaign score 0.05
    Feature 4 variable previous score 0.04
    Feature 5 variable housing_no score 0.03
    Feature 6 variable housing_yes score 0.02
    Feature 11 variable job_admin. score 0.01
    Feature 15 variable job_management score 0.01
    Feature 20 variable job_technician score 0.01
    Feature 23 variable education_secondary score 0.01
    Feature 24 variable education_tertiary score 0.01
    Feature 25 variable marital_divorced score 0.01
    Feature 28 variable sin_month score 0.09
    Feature 29 variable cos_month score 0.03
    Important features from decision treee are : 
    ['age', 'balance', 'duration', 'campaign', 'previous', 'housing_no', 'housing_yes', 'job_admin.', 'job_management', 'job_technician', 'education_secondary', 'education_tertiary', 'marital_divorced', 'sin_month', 'cos_month']

![](Summary_files/figure-html/cell-42-output-2.png)

Features selected from this algorithm are

*   Age
*   Balance
*   Duration
*   Campaign
*   Previous
*   Housing
*   Job
*   Education
*   Marital
*   Month - Sin,cos

We have all the important features from EDA here

10.2 Hyperparameter tuning
--------------------------

For tuning the hyperparameter’s we will use GridSearch CV.

    Fitting 5 folds for each of 168 candidates, totalling 840 fits

    Best parameters from Grid Search CV : 
    {'criterion': 'entropy', 'max_depth': 6, 'max_features': 0.8, 'splitter': 'best'}

Training model based on the parameters we got from Grid SearchCV.

    0.8935468676401319
    [[7104  413]
     [ 491  484]]
                  precision    recall  f1-score   support
    
               0       0.94      0.95      0.94      7517
               1       0.54      0.50      0.52       975
    
        accuracy                           0.89      8492
       macro avg       0.74      0.72      0.73      8492
    weighted avg       0.89      0.89      0.89      8492
    

From the decision tree we have better precision, recall, accuracy and thus better f1 score. Hence, decision tree is performing better than logistic regression.

AUC Curve : [Figure 1](#fig-roc-curve)  
Precision Recall Curve : [Figure 2](#fig-pr-curve)

11 Random Forest
================

11.1 Feature Selection
----------------------

    Important features from random forest :
    ['age', 'balance', 'duration', 'campaign', 'previous', 'housing_no', 'housing_yes', 'job_admin.', 'job_management', 'job_technician', 'education_secondary', 'education_tertiary', 'marital_married', 'marital_single', 'sin_month', 'cos_month']

![](Summary_files/figure-html/cell-45-output-2.png)

11.2 Hyperparameter Tuning
--------------------------

    Fitting 3 folds for each of 32 candidates, totalling 96 fits

    {'bootstrap': True, 'max_depth': 90, 'max_features': 3, 'n_estimators': 300}

    Training accuracy 1.0
    Testing set accuracy 0.8986104569006124
    [[7273  244]
     [ 617  358]]
                  precision    recall  f1-score   support
    
               0       0.92      0.97      0.94      7517
               1       0.59      0.37      0.45       975
    
        accuracy                           0.90      8492
       macro avg       0.76      0.67      0.70      8492
    weighted avg       0.88      0.90      0.89      8492
    

We are getting best performance from Random Forest but we are not sure why we are getting such idealistic results so we would also apply cross validation to test our results

    {'Training Accuracy scores': array([1., 1., 1., 1., 1.]),
     'Mean Training Accuracy': 100.0,
     'Training Precision scores': array([1., 1., 1., 1., 1.]),
     'Mean Training Precision': 1.0,
     'Training Recall scores': array([1., 1., 1., 1., 1.]),
     'Mean Training Recall': 1.0,
     'Training F1 scores': array([1., 1., 1., 1., 1.]),
     'Mean Training F1 Score': 1.0,
     'Validation Accuracy scores': array([0.89549603, 0.90035325, 0.89696791, 0.89842485, 0.89842485]),
     'Mean Validation Accuracy': 89.79333779716873,
     'Validation Precision scores': array([0.58850575, 0.61956522, 0.58969072, 0.59958506, 0.61009174]),
     'Mean Validation Precision': 0.6014876983054311,
     'Validation Recall scores': array([0.3252859 , 0.36213469, 0.36340534, 0.36768448, 0.33842239]),
     'Mean Validation Recall': 0.35138655828976595,
     'Validation F1 scores': array([0.41898527, 0.45709703, 0.44968553, 0.45583596, 0.43535188]),
     'Mean Validation F1 Score': 0.4433911363649415}

After applying cross validation, we are getting some what real estimates.

AUC Curve : [Figure 1](#fig-roc-curve)  
Precision Recall Curve : [Figure 2](#fig-pr-curve)

12 Linear SVC
=============

Finding a linear hyperplane that tries to separate two classes.

    0.8906029203956665
    [[7413  104]
     [ 825  150]]
                  precision    recall  f1-score   support
    
               0       0.90      0.99      0.94      7517
               1       0.59      0.15      0.24       975
    
        accuracy                           0.89      8492
       macro avg       0.75      0.57      0.59      8492
    weighted avg       0.86      0.89      0.86      8492
    

13 SVC
======

Finding a complex hyperplane that tries to separate the classes.

    0.8922515308525671
    [[7446   71]
     [ 844  131]]
                  precision    recall  f1-score   support
    
               0       0.90      0.99      0.94      7517
               1       0.65      0.13      0.22       975
    
        accuracy                           0.89      8492
       macro avg       0.77      0.56      0.58      8492
    weighted avg       0.87      0.89      0.86      8492
    

14 Naive Bayes
==============

Naive Bayes a naive assumption that all the features are independent of each other and thus by reducing the complexity of computing conditional probabilities it evaluates the probability of 0 and 1.

    Fitting 10 folds for each of 100 candidates, totalling 1000 fits

    GaussianNB(var_smoothing=0.0533669923120631)
    Model score is 0.8871879415920867

![](Summary_files/figure-html/cell-51-output-3.png)

    test set evaluation: 
    0.8871879415920867
    [[7291  226]
     [ 732  243]]
                  precision    recall  f1-score   support
    
               0       0.91      0.97      0.94      7517
               1       0.52      0.25      0.34       975
    
        accuracy                           0.89      8492
       macro avg       0.71      0.61      0.64      8492
    weighted avg       0.86      0.89      0.87      8492
    

14.1 For balanced
-----------------

For balanced dataset, as we can see there is a slight improvement in performance. The f1 score has improved and also, the yellow bars are now slightly shifted towards right side.

    Model score is 0.459609043805935

![](Summary_files/figure-html/cell-52-output-2.png)

    test set evaluation: 
    0.459609043805935
    [[2994 4523]
     [  66  909]]
                  precision    recall  f1-score   support
    
               0       0.98      0.40      0.57      7517
               1       0.17      0.93      0.28       975
    
        accuracy                           0.46      8492
       macro avg       0.57      0.67      0.42      8492
    weighted avg       0.89      0.46      0.53      8492
    

As we can see from the graph for the red and yellow bars for yes(1 term deposit) are coming on the opposite sides which is not expected.

AUC Curve : [Figure 1](#fig-roc-curve)  
Precision Recall Curve : [Figure 2](#fig-pr-curve)

15 KNN
======

Using the k - nearest neighbours we try to predict the testing dataset. Now to find the optimal k value we will look into precision and accuracy curve for different k values.

    Maximum accuracy:- 0.8936646255299106 at K = 16

![](Summary_files/figure-html/cell-54-output-2.png)

Accuracy curve for different k values

    Maximum Precision:- 0.23186915390816643 at K = 6

![](Summary_files/figure-html/cell-55-output-2.png)

Precision curve for different k values

Based on the above plot, optimal k value is 3, with maximum f1 score of 0.64.

    Train set accuracy 0.9290508714083844
    Test set accuracy 0.8795336787564767
    [[7202  315]
     [ 708  267]]
                  precision    recall  f1-score   support
    
               0       0.91      0.96      0.93      7517
               1       0.46      0.27      0.34       975
    
        accuracy                           0.88      8492
       macro avg       0.68      0.62      0.64      8492
    weighted avg       0.86      0.88      0.87      8492
    

AUC Curve : [Figure 1](#fig-roc-curve)  
Precision Recall Curve : [Figure 2](#fig-pr-curve)

16 ROC -AUC Curve
=================

![](Summary_files/figure-html/fig-roc-curve-output-1.png)

Figure 1: AUC ROC Curve for all Models

17 Precision Recall Curve
=========================

In imbalance problem since we have a high number of Negatives, this makes the False Posiitve Rate as low, resulting in the shift of ROC AUC Curve towards left, which is slightly misleading.

So in imbalance problem we usually make sure to look at precision recall curve as well.

![](Summary_files/figure-html/fig-pr-curve-output-1.png)

Figure 2: Precision Recall Curve for all Models

As per the ROC Curve and Precision Recall curve, KNN is performing best. But after combining these results with precision recall curve, we suggest using Random Forest for our problem.

18 Summary
==========

Table 1: Summary of Models

Model

Accuracy

Precision

Recall

AUC

Logistic(Cutoff=0.25)

0.88

0.51

0.58

0.872

Logistic (Balanced-Train)

0.85

0.49

0.54

Decision Tree

0.91

0.66

0.47

0.923

Random Forest

0.88

0.66

0.46

0.913

SVC

0.89

0.75

0.15

Linear SVC

0.89

0.62

0.16

Gaussian Bayes

0.88

0.50

0.25

0.841

KNN

0.92

0.78

0.54

0.965

Naive Bayes

0.85

0.56

0.02

Naive Bayes (Balanced-Train)

0.69

0.19

0.35

See [Table 1](#tbl-letters).

19 Conclusion
=============

Our model would be beneficial in the following ways :

*   For target marketing for bank campaigns, or in other events. For example based on the customer’s job, age and loan history the model would can easily predict whether the customer is going to subscribe to the term deposit or not. So out of the million people, we can easily shortlist people based on our model and spend the time on them so as to improve efficiency.
    
*   Improving buissness effficiency of banks. Since using the eda or model we can easily check the subscription insights, it would be very helpful for banks to improve their stratergies. For example, based on the monthly subscription rates, if banks are deciding the campaign promotion time, it can improve there efficiency.
    
*   Since, we have month as a input factor in our model, and all other values are static, we can even find the best month to contact customer based on the predicted probability of the customer. As there can be a relation between the job type and the month they are subscribing or their fluctuating balance and age. This can be very useful in finding the best time to contact.
    
*   Based on the model, since the number of contact is playing a major role, if we have the optimal time to contact them, we can restrict our calls to less than 5 and find a better turnover.
    
*   We didn’t see any relation with the social and economic factors here, but if we had the data for multiple years, there was a possibility of finding a relation. Our model can accomodate these factors as well, and if trained by accomodating these factors as well, this can be helpful for banks in finding the proper time for there campaign.
    

Hence, analyzing this kind of marketing dataset has given us valuable insight into how we can tweak our model to give buisness insights as well as customer insights to improve subscription of term deposits.

20 Reference
============

*   https://www.kaggle.com/janiobachmann/bank-marketing-dataset
    
*   (PDF) Data Analysis of a Portuguese marketing campaign using bank … (no date). Available at: https://www.researchgate.net/publication/339988208\_Data\_Analysis\_of\_a\_Portuguese\_Marketing\_Campaign\_using\_Bank\_Marketing\_data\_Set (Accessed: December 20, 2022).
    
*   Bank marketing data set. (n.d.). 1010data.com. Retrieved December 20, 2022, from https://docs.1010data.com/Tutorials/MachineLearningExamples/BankMarketingDataSet\_2.html
    
*   Manda, H., Srinivasan, S., & Rangarao, D. (2021). IBM Cloud Pak for Data: An enterprise platform to operationalize data, analytics, and AI. Packt Publishing.
    
*   Solving Bank Marketing Calssification Problem - Databricks. (n.d.). Databricks.com. Retrieved December 20, 2022, from https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/8143187682226564/2297613386094950/3186001515933643/latest.html
    
*   Solving Bank Marketing Calssification Problem - Databricks. (n.d.). Databricks.com. Retrieved December 20, 2022, from https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/8143187682226564/2297613386094950/3186001515933643/latest.html
    
*   Bank Marketing Data Set. (n.d.). UCI Machine Learning Repository. Retrieved December 20, 2022, from https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
    
*   https://tradingeconomics.com/
    

window.document.addEventListener("DOMContentLoaded", function (event) { const toggleBodyColorMode = (bsSheetEl) => { const mode = bsSheetEl.getAttribute("data-mode"); const bodyEl = window.document.querySelector("body"); if (mode === "dark") { bodyEl.classList.add("quarto-dark"); bodyEl.classList.remove("quarto-light"); } else { bodyEl.classList.add("quarto-light"); bodyEl.classList.remove("quarto-dark"); } } const toggleBodyColorPrimary = () => { const bsSheetEl = window.document.querySelector("link#quarto-bootstrap"); if (bsSheetEl) { toggleBodyColorMode(bsSheetEl); } } toggleBodyColorPrimary(); const icon = ""; const anchorJS = new window.AnchorJS(); anchorJS.options = { placement: 'right', icon: icon }; anchorJS.add('.anchored'); const clipboard = new window.ClipboardJS('.code-copy-button', { target: function(trigger) { return trigger.previousElementSibling; } }); clipboard.on('success', function(e) { // button target const button = e.trigger; // don't keep focus button.blur(); // flash "checked" button.classList.add('code-copy-button-checked'); var currentTitle = button.getAttribute("title"); button.setAttribute("title", "Copied!"); setTimeout(function() { button.setAttribute("title", currentTitle); button.classList.remove('code-copy-button-checked'); }, 1000); // clear code selection e.clearSelection(); }); function tippyHover(el, contentFn) { const config = { allowHTML: true, content: contentFn, maxWidth: 500, delay: 100, arrow: false, appendTo: function(el) { return el.parentElement; }, interactive: true, interactiveBorder: 10, theme: 'quarto', placement: 'bottom-start' }; window.tippy(el, config); } const noterefs = window.document.querySelectorAll('a\[role="doc-noteref"\]'); for (var i=0; i<noterefs.length; i++) { const ref = noterefs\[i\]; tippyHover(ref, function() { // use id or data attribute instead here let href = ref.getAttribute('data-footnote-href') || ref.getAttribute('href'); try { href = new URL(href).hash; } catch {} const id = href.replace(/^#\\/?/, ""); const note = window.document.getElementById(id); return note.innerHTML; }); } var bibliorefs = window.document.querySelectorAll('a\[role="doc-biblioref"\]'); for (var i=0; i<bibliorefs.length; i++) { const ref = bibliorefs\[i\]; const cites = ref.parentNode.getAttribute('data-cites').split(' '); tippyHover(ref, function() { var popup = window.document.createElement('div'); cites.forEach(function(cite) { var citeDiv = window.document.createElement('div'); citeDiv.classList.add('hanging-indent'); citeDiv.classList.add('csl-entry'); var biblioDiv = window.document.getElementById('ref-' + cite); if (biblioDiv) { citeDiv.innerHTML = biblioDiv.innerHTML; } popup.appendChild(citeDiv); }); return popup.innerHTML; }); } });

}); } var bibliorefs = window.document.querySelectorAll('a\[role="doc-biblioref"\]'); for (var i=0; i
