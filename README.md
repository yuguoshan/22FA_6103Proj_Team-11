# ANANALYSISOFPORTUGUESEBANKMARKETINGDATA

## TheGeorgeWashingtonUniversity(DATS 6103 :AnIntroductionto

## DataMining)

```
TEAM 11 :AnjaliMudgal,GuoshanYu,MedhaswetaSen
```
```
December 20 , 2022
```
## INTRODUCTION

Bankmarketingisthepracticeofattractingandacquiringnewcustomersthrough
traditionalmediaanddigitalmediastrategies.Theuseofthesemediastrategies
helpsdeterminewhatkindofcustomerisattractedtoacertaininstitutions.This
alsoincludesdifferentbankinginstitutionspurposefullyusingdifferentstrategiesto
attractthetypeofcustomertheywanttodobusinesswith.

Marketinghasevolvedfromacommunicationroletoarevenuegeneratingrole.The
consumerhasevolvedfrombeingapassiverecipientofmarketingmessagestoan
activeparticipantinthemarketingprocess.Technologyhasevolvedfrombeinga
meansofcommunicationtoameansofdatacollectionandanalysis.Dataanalytics
hasevolvedfrombeingameansofunderstandingtheconsumertoameansof
understandingtheconsumerandtheinstitution.

Bankmarketingstrategyisincreasinglyfocusedondigitalchannels,includingsocial
media,video,searchandconnectedTV.Asbankandcreditunionmarketersstriveto
promotebrandawareness,theyneedanewwaytoassesschannelROIandmore
accuratedatatoenablepersonalizedoffers.Addtothatthegrowingimportanceof
purpose-drivenmarketing.

Therelentlesspaceofdigitizationisdisruptingnotonlytheestablishedorderin
banking,butbankmarketingstrategies.Marketersatbothtraditionalinstitutions
anddigitaldisruptorsarefeelingthepressure.

Justasbankmarketersbegintomasteronechannel,consumersmovetoanother.
Manynowtogglebetweendevicesonaseeminglyinfinitenumberofplatforms,
makingitharderthaneverformarketerstopindowntherightconsumersatthe
righttimeintherightplace.


### TheDataSet

ThedatasetusedinthisanalysisisfromaPortuguesebank.Thedatasetcontains
41 , 188 observationsand 21 variables.Thevariablesincludethefollowing:

```
1.
```
- age(numeric)
2.


- job:typeofjob(categorical:‘admin.’,‘blue-
    collar’,‘entrepreneur’,‘housemaid’,‘management’,‘retired’,‘self-
    employed’,‘services’,‘student’,‘technician’,‘unemployed’,‘unknown’)
3.
- marital:maritalstatus(categorical:
‘divorced’,‘married’,‘single’,‘unknown’;note:‘divorced’means
divorcedorwidowed)

4.

- education(categorical:
    ‘basic. 4 y’,‘basic. 6 y’,‘basic. 9 y’,‘high.school’,‘illiterate’,‘professional.cour
    se’,‘university.degree’,‘unknown’)

5.

- default:hascreditindefault?(categorical:‘no’,‘yes’,‘unknown’)

6.

- housing:hashousingloan?(categorical:‘no’,‘yes’,‘unknown’)

7.

- loan:haspersonalloan?(categorical:‘no’,‘yes’,‘unknown’)

8.

- contact:contactcommunicationtype(categorical:
    ‘cellular’,‘telephone’)

9.

- month:lastcontactmonthofyear(categorical:‘jan’,‘feb’,‘mar’,...,
    ‘nov’,‘dec’)

10.

- day_of_week:lastcontactdayoftheweek(categorical:
    ‘mon’,‘tue’,‘wed’,‘thu’,‘fri’)

11.

- duration:lastcontactduration,inseconds(numeric).Importantnote:
    thisattributehighlyaffectstheoutputtarget(e.g.,ifduration= 0 then
    y=‘no’).Yet,thedurationisnotknownbeforeacallisperformed.Also,
    aftertheendofthecallyisobviouslyknown.Thus,thisinputshould
    onlybeincludedforbenchmarkpurposesandshouldbediscardedif
    theintentionistohavearealisticpredictivemodel.

12.

- campaign:numberofcontactsperformedduringthiscampaignand
    forthisclient(numeric,includeslastcontact)

13.

- pdays:numberofdaysthatpassedbyaftertheclientwaslast
    contactedfromapreviouscampaign(numeric; 999 meansclientwas
    notpreviouslycontacted)

14.


- previous:numberofcontactsperformedbeforethiscampaignandfor
    thisclient(numeric)
15.
- poutcome:outcomeofthepreviousmarketingcampaign(categorical:
‘failure’,‘nonexistent’,‘success’)
16.
- emp.var.rate:employmentvariationrate-quarterlyindicator
(numeric)
17.
- cons.price.idx:consumerpriceindex-monthlyindicator(numeric)
18.
- cons.conf.idx:consumerconfidenceindex-monthlyindicator
(numeric)
19.
- euribor 3 m:euribor 3 monthrate-dailyindicator(numeric)
20.
- nr.employed:numberofemployees-quarterlyindicator(numeric)
21.
- balance-averageyearlybalance,ineuros(numeric)
22.
- y-hastheclientsubscribedatermdeposit?(binary:‘yes’,‘no’)

### TheSMARTQuestions

TheSMARTquestionsareasfollows:

1 .Relationshipbetweensubscribingthetermdepositandhowmuchthecustomeris
contacted(lastcontact,Campaign,Pdays,PreviousNumberofcontacts)


```
2. Findoutthefinanciallystablepopulation?Willthataffecttheoutcome?
```
3 .Effectofdimensionalityreductiononaccuracyofthemodel.

```
4. Howarethelikelihoodofsubscriptionsaffectedbysocialandeconomic
factors?
```
Throughoutthepaperwewouldtrytoanswerthequestions

Importingtherequiredlibraries

### Importingthedataset

### BasicInformationaboutthedata

Shape of dataset is : ( 45211 , 23 )
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 45211 entries, 0 to 45210
Data columns (total 23 columns):
# Column Non-Null Count Dtype

- -- ------ -------------- -----
    0 age 45211 non-null int 64
    1 job 45211 non-null object
    2 marital 45211 non-null object
    3 education 45211 non-null object
    4 default 45211 non-null object
    5 balance 45211 non-null int 64
    6 housing 45211 non-null object
    7 loan 45211 non-null object
    8 contact 45211 non-null object
    9 day 45211 non-null int 64
    10 month 45211 non-null object
    11 duration 45211 non-null int 64
    12 campaign 45211 non-null int 64
    13 pdays 45211 non-null int 64
    14 previous 45211 non-null int 64
    15 poutcome 45211 non-null object
    16 y 45211 non-null int 64
    17 month_int 45211 non-null int 64
    18 cons.conf.idx 45211 non-null float 64
    19 emp.var.rate 45211 non-null float 64
    20 euribor 3 m 45211 non-null float 64
    21 nr.employed 45211 non-null float 64
    22 cons.price.idx 45211 non-null float 64
dtypes: float 64 ( 5 ), int 64 ( 9 ), object( 9 )
memory usage: 7. 9 + MB
Columns in dataset
    None


## ExploratoryDataAnalysis(EDA)

### Distributionofy(target)variable

Wehave 45 , 211 datapoints,ifourmodelpredictsonly 0 asoutput,wewouldstill
get 88 %accuracy,soourdatasetisunbalancedwhichmaygivesmisleadingresults.
Alongwiththeaccuracy,wewillalsoconsiderprecisionandrecallforevaluation.

### MissingvaluesandOutliers

**Education**

Here,eventhoughwedonothaveanymissingvaluesbutwehave‘unknown’and
‘other’ascategories,sowewillfirstgetridofthem.Thevariableswith‘unknown’
rowsareEducationandContactshownedbelow.


Text( 0. 5 , 1. 0 , 'Type of education Distribution')

**Contact**
Text( 0. 5 , 1. 0 , 'Type of Contact Distribution')


- sincethetypeofcommunication(cellularandtelephone)isnotreallyagood
    indicatorofsubcription,wedropthisvariable.

**Poutcome**

poutcome
failure 4901
other 1840
success 1511
unknown 36959
dtype: int 64

Thereare _36959 unknown_ values( 82 %)and 1840 valueswithother( 4. 07 %)
category,wewilldirectlydropthesecolumns.


### Outliers

- Thereareoutliersindurationandbalancesoweneedtogetridofthem.

## DataCleaning

- Contactisnotusefulsowedropit.
- Inpoutcome,wehavealotof‘unknown’and‘other’valuessowedropit.
- Dayisnotgivinganyrelevantinfomationsowedropit.
- Removingtheunknownsfrom‘job’and‘education’columns.
- Removetheoutliersfrombalanceandduration.

### Droppingtheirrelavantcolumnsandmissingvalues

for job
unknown : 288
dropping rows with value as unknown in job
for education
unknown : 1730
dropping rows with value as unknown in education

### Outlierremoval

Wehaveoutliersinbalanceandduration,sotogetridofthemwewouldtryto
removetheenteriesfewstandarddeviationaway,sincefromthehistogramsmost


oftheenteriesarearoundmeanonly,weareremovingtheenteriesmorethan 3 SD
away.

**_Balance-Outliers_**
removing entries before balance - 7772. 283533
dtype: float 64 and after balance 10480. 338218
dtype: float 64

**_Duration-Outliers_**

Droppingrowswherethedurationofcallsislessthan 5 secsincethatisirrelevant.
Andalsosinceconvertingthecalldurationinminutesratherthansecondsmakes
moresensewewouldconvertitintominutes.

plottingviolenplotfordurationandbalanceaftercleaningdata

## DataVisualization

Let’visualizeimportantrelationshipsbetweenvariablesnow.

### SMARTQuestion 1 :

Relationshipbetweensubscribingthetermdepositandhowmuchthecustomeris
contacted(lastcontact,Campaign,Pdays,PreviousNumberofcontacts)


Answer:Basedonlastcontactinfoonlynumberofcontactsperformedduringthis
campaigniscontributingalottowardssubscriptionrates.

Suggestion:Peoplewhoarecontactedlessthan 5 timesshouldbetargetedmore.
Also,theycouldcontactinlessfrequencyinordertoattractmoretargetcustomers.
Theplotbelowshowstherelationshipbetweenthenumberofcallsanddurationvs
subscription

**NumberofcallsversusDurationandaffectonsubscription**

Hereifwenotice,peoplearemorelikelytosubscribeifthenumberofcallsareless
than 5.

Checkingbetweenpdaysandpreviousaswell

Hereaswecanseefromthet-test,t

```
13.
```

- pdays:numberofdaysthatpassedbyaftertheclientwaslast
    contactedfromapreviouscampaign(numeric; 999 meansclientwas
    notpreviouslycontacted)
14.
- previous:numberofcontactsperformedbeforethiscampaignandfor
thisclient(numeric)

Wecannoticefromtheplotthatthereisnorelationshipbetweensubscriptionwith
pdaysorprevious.Thedatapointsaredistrubutedrandomlyalongtheaxies.

### Monthwisesubscription

Text( 0. 5 , 0 , 'Month')


MaximumpercentageofpeoplehavesubscribedinthemonthofMarchbutbankis
contactingpeoplemoreinthemonthofMay.

**Suggestion** :Soit’sbettertocontactcustomer’sbasedonthesubcriptionrateplot.

**SMARTQuestion 7 :Howarethelikelihoodofsubscriptionsaffectedbysocialand
economicfactors?**
month cons.conf.idx emp.var.rate euribor 3 m nr.employed
0 jan 1310 1310 1310 1310
1 feb 2492 2492 2492 2492
2 mar 439 439 439 439
3 apr 2772 2772 2772 2772
4 may 13050 13050 13050 13050
5 jun 4874 4874 4874 4874
6 jul 6550 6550 6550 6550
7 aug 5924 5924 5924 5924
8 sep 514 514 514 514
9 oct 661 661 661 661
10 nov 3679 3679 3679 3679
11 dec 195 195 195 195

**Answer** :Basedontheabovetablewecanseethatthereisnodistinguishable
differenceinthemonthofmarchormayfromrestofallthemonth,sosocialand
economicfactor **donothavemajorinfluence** ontheoutcome.

**SMARTQuestion 2**

Findoutthe **financiallystable** population?Willthataffecttheoutcome?


Wewilltrytofindthefinanciallystablepopulationbasedonage,jobs,loanand
balance.

**Loan**
Text( 0. 5 , 1. 0 , 'Type of loan Distribution')

Text( 0. 5 , 1. 0 , 'Type of housing Distribution')


Peoplewithhousingloansarelesslikelytosubscribetotermdepositbutthe
differencehereisnothuge.

Text( 0. 5 , 1. 0 , 'Type of default Distribution')


Sopeoplewhohavenotpaidbackthereloansandhavecredits,havenotsubcribed
tothetermdeposit.

- peoplewhohaveloansaresubscribingtotermdepositless.

**Age**

Elderpeoplemightbemorefinanciallystablesincetheyaresubscripedtotheterm
depositmore.

- Peoplewhoareoldaremorelikelytosubscribetotermdeposit.


**Job**

Peopleinbluecollarandmanagementjobsarecontactedmore,whichshouldnotbe
thecase.Sincetheyhavelesssubscriptionrates.Unlikepopularassumption,


students,retiredandunemploymentseemtohaveahighsubscriptionrates.Even
thoughtheyarecontactedveryless.

**suggestion** :Thehighsubscriptedrategroup(students,retiredandunemployment)
shouldbecontactedmore.

**Balance**

Checkingthesubscriptionsineachbalancegroups

balGroup % Contacted % Subscription
0 low balance 60. 339143 10. 503513
1 moderate balance 17. 399906 14. 036275
2 high balance 13. 709374 16. 715341
3 Negative 8. 551578 5. 700909
balanceGroup Contact Rate Subscription Rate
0 Negative 8. 551578 5. 700909
1 low balance 60. 339143 10. 503513
2 moderate balance 17. 399906 14. 036275
3 high balance 13. 709374 16. 715341


**suggestion** :Peoplewithmoderatetohighbalance,arecontactedlessbuttheyhave
highsubscriptionratessobankshouldtargetthemmore.

Itmightbepossiblethatbalancegroupandjobsaretellingthesameinformation
sincesomejobsmighthavehighsalaryandthusbalancegroupsmightbedepicting
jobsonly,sowewilltrytolookatthemtogether.

BalanceGroupversusJob

Text( 0. 5 , 1. 0 , 'Contact for each balance group in job category')


StudentandRetiredaremorelikelytosubscribeandusuallyhavemoderatetohigh
balance.

Wefoundfromthesecondbarchartthatonlythelowbalancegroupsaretargetedin
eachcategoryeventhoughmoderatetohighbalancecategoryaremorelikelyto
subscribe.

## DataEncoding

### OneHotEncoding

Wewouldencode‘housing’,‘loan’,‘default’,‘job’,‘education’and‘marital’astheyare
allcategoricalvariables.

### Sin-Cosencoding

Transformingmonthintosinandcossothattherecyclicnature(jan-decareasclose
asjan-feb)isretainedwhichisusuallylostinlabelencoding.Unlikeonehot
encoding,thedimensionwillreducefrom 12 (month_jan,month_feb...month_dec)
to 2 (sin_month,cos_month)

<AxesSubplot: xlabel='sin_month', ylabel='cos_month'>


### Droppingunnecessarycolumnsirrelevantformodelling

Herewedroppedthe‘month’columnastheyareencoded.Also,wedropped
irrelvantvariables‘pdays’andenconomicfactors(‘cons.conf.idx’,‘emp.var.rate’,
‘euribor 3 m’,‘nr.employed’,‘cons.price.idx’)formodelling.

## DataModeling

### SplittingourDataset

Wearesplittingourdatasetin 1 : 4 ratiofortrainingandtestingset.

### BalancingOurDataset

Wetriedtobalanceourdatasetusingfollowingmethods:

- UpsamplingusingSMOTE
- Sinandcostransformationfrommonth_int.

## Scalingnumericvariables

Scalingage,balance,durationsothatouralgorithmsperformbetterandall
variablesaretreatedequally.Sinceallthreevariablesareindifferentscales,sowe
transformthemintosamestandard.


## LogisticRegression

PerformingLogisticRegressiononbothbalancedandunbalanceddataset.RFEis
usedinselectingthemostimportantfeatures##UnbalancedDataset

Columns selected by RE ['duration', 'housing_no', 'housing_yes', 'loan_
no', 'loan_yes', 'job_admin.', 'job_blue-collar', 'job_entrepreneur', '
job_housemaid', 'job_retired', 'job_self-employed', 'job_student', 'edu
cation_primary', 'education_tertiary', 'cos_month', 'age', 'balance', '
sin_month']

AswecanseefromRFE,themostrelevantfeaturesare:

- Duration
- Housing
- Loan
- Job
- Education
- cos_month

FromotherfeaturesselectiontechniquesandEDA,wecanseethat‘age’and
‘balance’alsocontrubutedtothesubscrption,soweaddedupthesevariablesas
well.

Applyingmodelwithselectedfeatures

Accuracy for training set 0. 8918982571832312
Accuracy for testing set 0. 884950541686293
Confusion matrix
[[ 7335 150 ]
[ 827 180 ]]
precision recall f 1 - score support

```
0 0. 90 0. 98 0. 94 7485
1 0. 55 0. 18 0. 27 1007
```
accuracy 0. 88 8492
macro avg 0. 72 0. 58 0. 60 8492
weighted avg 0. 86 0. 88 0. 86 8492

Here,theaccuracyis 89 %buttheprecision( 0. 59 )andrecallratevalue( 0. 20 )islow.
Andwealsocheckonthebalanceddatasetsincethelowrecallratemightbecaused
becauseofthelessnumberofy= 1 value.

## BalancedDataset

Columns selected by RE ['housing_yes', 'loan_yes', 'job_blue-collar', '
job_entrepreneur', 'job_housemaid', 'job_management', 'job_self-employe
d', 'job_services', 'job_technician', 'job_unemployed', 'education_prim


ary', 'education_secondary', 'marital_divorced', 'marital_married', 'ma
rital_single']

Accuracy for training set 0. 8830944224565138
Accuracy for testing set 0. 8224211022138483
Confusion matrix
[[ 6328 1157 ]
[ 351 656 ]]
precision recall f 1 - score support

```
0 0. 95 0. 85 0. 89 7485
1 0. 36 0. 65 0. 47 1007
```
accuracy 0. 82 8492
macro avg 0. 65 0. 75 0. 68 8492
weighted avg 0. 88 0. 82 0. 84 8492

Here,importantfeaturesare*Housing*Loan*Job*Education*MaritalStatus

Wealsoaddedtheimportantfeaturesfromunbalaceddataset*Duration*Age*
Month*Balance

Hereeventhoughtheprecisionandrecallhaveimproved,andaccuracyhasdropped
down,buttheimportantrelationshipsarelostsincethetrainingdatanowis
artificiallygenerateddatapoints.Wewilltrytofindtheoptimalcut-offvaluefor
originaldatasetandcompareitwiththemodelforbalanceddata.

**Decidingcutoffvalueforlogisticregression-Unbalance**

Buttohavegoodvaluesforcut-offwewouldtrytofindacutoffwheretheprecision
andrecallvaluesaredecent

Based on plot we would choose 0. 25 as cut off
Accuracy for testing set 0. 8784738577484692
Confusion matrix
[[ 7016 469 ]
[ 563 444 ]]
precision recall f 1 - score support

```
0 0. 93 0. 94 0. 93 7485
1 0. 49 0. 44 0. 46 1007
```
accuracy 0. 88 8492
macro avg 0. 71 0. 69 0. 70 8492
weighted avg 0. 87 0. 88 0. 88 8492


_OptimalCutoffat 0. 25_

Hereasafterapplyingfeatureselection,findingoptimizedcut-off,weareableto
achievehigheraccuracywithoptimalprecisionandrecall.Resultingfromthe
comparison,wewouldcontinueourmodellingswithunbalancedataset.

**SmartQuestion 5 :Theoptimalcutoffvalueforclassificationofourimbalancedataset.**

**Answer** :Theoptimalcutoffvalueforourimbalancedatasetis 0. 25 astheprecision-
recallchartindicated.

**SMARTQuestion 2 :Sincethedatasetisimbalanced,willdownsampling/upsampling
orothertechniquesimproveupontheaccuracyofmodels.**

**Answer** :Asobservedfromabovethereisaslightimprovementinaccuracy,
precisionandrecallafterweapplySMOTE,butthatimprovementcanalsobe
acheivedbyadjustingthecutoffvalueaswell.So,weshouldalwaystryadjusting
cut-offfirst,beforeupsampling.

ForROC-AUCcurverefer(Figure 1 ).
Forprecisionrecallcurverefer(Figure 2 ).


## DecisionTree

### FeatureSelection

Feature 0 variable age score 0. 12
Feature 1 variable balance score 0. 16
Feature 2 variable duration score 0. 33
Feature 3 variable campaign score 0. 04
Feature 4 variable previous score 0. 05
Feature 5 variable housing_no score 0. 01
Feature 6 variable housing_yes score 0. 04
Feature 12 variable job_blue-collar score 0. 01
Feature 15 variable job_management score 0. 01
Feature 20 variable job_technician score 0. 01
Feature 28 variable sin_month score 0. 08
Feature 29 variable cos_month score 0. 03
Important features from decision treee are :
['age', 'balance', 'duration', 'campaign', 'previous', 'housing_no', 'h
ousing_yes', 'job_blue-collar', 'job_management', 'job_technician', 'si
n_month', 'cos_month']

Featuresselectedfromthisalgorithmare

- Age
- Balance
- Duration
- Campaign
- Previous
- Housing


- Job
- Education
- Marital
- Month-Sin,cos

WehavealltheimportantfeaturesfromEDAhere

### Hyperparametertuning

Fortuningthehyperparameter’swewilluseGridSearchCV.

Fitting 5 folds for each of 168 candidates, totalling 840 fits

Best parameters from Grid Search CV :
{'criterion': 'entropy', 'max_depth': 6 , 'max_features': None, 'splitte
r': 'best'}

TrainingmodelbasedontheparameterswegotfromGridSearchCV.

0. 8916627414036741
[[ 7176 309 ]
[ 611 396 ]]
precision recall f 1 - score support

```
0 0. 92 0. 96 0. 94 7485
1 0. 56 0. 39 0. 46 1007
```
accuracy 0. 89 8492
macro avg 0. 74 0. 68 0. 70 8492
weighted avg 0. 88 0. 89 0. 88 8492

Fromthedecisiontreewehavebetterprecision,recall,accuracyandthusbetterf 1
score.Hence,decisiontreeisperformingbetterthanlogisticregression.

AUCCurve:Figure 1
PrecisionRecallCurve:Figure 2

## RandomForest

### FeatureSelection

Important features from random forest :
['age', 'balance', 'duration', 'campaign', 'previous', 'housing_no', 'h
ousing_yes', 'job_admin.', 'job_management', 'job_technician', 'educati
on_secondary', 'education_tertiary', 'marital_married', 'marital_single
', 'sin_month', 'cos_month']


### HyperparameterTuning

Fitting 3 folds for each of 32 candidates, totalling 96 fits

{'bootstrap': True, 'max_depth': 110 , 'max_features': 3 , 'n_estimators':
1000 }

Training accuracy 1. 0
Testing set accuracy 0. 8951954780970325
[[ 7243 242 ]
[ 648 359 ]]
precision recall f 1 - score support

```
0 0. 92 0. 97 0. 94 7485
1 0. 60 0. 36 0. 45 1007
```
accuracy 0. 90 8492
macro avg 0. 76 0. 66 0. 69 8492
weighted avg 0. 88 0. 90 0. 88 8492

WearegettingbestperformancefromRandomForestbutwearenotsurewhywe
aregettingsuchidealisticresultssowewouldalsoapplycrossvalidationtotestour
results

{'Training Accuracy scores': array([ 1 ., 1 ., 1 ., 1 ., 1 .]),
'Mean Training Accuracy': 100. 0 ,
'Training Precision scores': array([ 1 ., 1 ., 1 ., 1 ., 1 .]),
'Mean Training Precision': 1. 0 ,
'Training Recall scores': array([ 1 ., 1 ., 1 ., 1 ., 1 .]),
'Mean Training Recall': 1. 0 ,


'Training F 1 scores': array([ 1 ., 1 ., 1 ., 1 ., 1 .]),
'Mean Training F 1 Score': 1. 0 ,
'Validation Accuracy scores': array([ 0. 90241389 , 0. 8971151 , 0. 8978510
5 , 0. 89665832 , 0. 90328279 ]),
'Mean Validation Accuracy': 89. 94642314134781 ,
'Validation Precision scores': array([ 0. 62526767 , 0. 58672377 , 0. 594360
09 , 0. 57677165 , 0. 64009112 ]),
'Mean Validation Precision': 0. 6046428582347663 ,
'Validation Recall scores': array([ 0. 37435897 , 0. 35128205 , 0. 35083227 ,
0. 37564103 , 0. 36025641 ]),
'Mean Validation Recall': 0. 3624741455727371 ,
'Validation F 1 scores': array([ 0. 46832398 , 0. 43945469 , 0. 44122383 , 0. 4
5496894 , 0. 46103363 ]),
'Mean Validation F 1 Score': 0. 4530010159118049 }

Afterapplyingcrossvalidation,wearegettingsomewhatrealestimates.

AUCCurve:Figure 1
PrecisionRecallCurve:Figure 2

## LinearSVC

Findingalinearhyperplanethattriestoseparatetwoclasses.

0. 8857748469147433
[[ 7381 104 ]
[ 866 141 ]]
precision recall f 1 - score support

```
0 0. 89 0. 99 0. 94 7485
1 0. 58 0. 14 0. 23 1007
```
accuracy 0. 89 8492
macro avg 0. 74 0. 56 0. 58 8492
weighted avg 0. 86 0. 89 0. 85 8492

## SVC

Findingacomplexhyperplanethattriestoseparatetheclasses.

0. 8865991521431936
[[ 7423 62 ]
[ 901 106 ]]
precision recall f 1 - score support

```
0 0. 89 0. 99 0. 94 7485
1 0. 63 0. 11 0. 18 1007
```

accuracy 0. 89 8492
macro avg 0. 76 0. 55 0. 56 8492
weighted avg 0. 86 0. 89 0. 85 8492

## NaiveBayes

NaiveBayesanaiveassumptionthatallthefeaturesareindependentofeachother
andthusbyreducingthecomplexityofcomputingconditionalprobabilitiesit
evaluatestheprobabilityof 0 and 1.

Fitting 10 folds for each of 100 candidates, totalling 1000 fits

GaussianNB(var_smoothing= 0. 0533669923120631 )
Model score is 0. 886481394253415

test set evaluation:
0. 886481394253415
[[ 7293 192 ]
[ 772 235 ]]
precision recall f 1 - score support

```
0 0. 90 0. 97 0. 94 7485
1 0. 55 0. 23 0. 33 1007
```
accuracy 0. 89 8492
macro avg 0. 73 0. 60 0. 63 8492
weighted avg 0. 86 0. 89 0. 87 8492


### Forbalanced

Forbalanceddataset,aswecanseethereisaslightimprovementinperformance.
Thef 1 scorehasimprovedandalso,theyellowbarsarenowslightlyshiftedtowards
rightside.

Model score is 0. 4401789919924635

test set evaluation:
0. 4401789919924635
[[ 2818 4667 ]
[ 87 920 ]]
precision recall f 1 - score support

```
0 0. 97 0. 38 0. 54 7485
1 0. 16 0. 91 0. 28 1007
```
accuracy 0. 44 8492
macro avg 0. 57 0. 65 0. 41 8492
weighted avg 0. 87 0. 44 0. 51 8492

Aswecanseefromthegraphfortheredandyellowbarsforyes( 1 termdeposit)are
comingontheoppositesideswhichisnotexpected.

AUCCurve:Figure 1
PrecisionRecallCurve:Figure 2

## KNN

Usingthek-nearestneighbourswetrytopredictthetestingdataset.Nowtofind
theoptimalkvaluewewilllookintoprecisionandaccuracycurvefordifferentk
values.


Maximum accuracy:- 0. 8888365520489873 at K = 36

_Accuracycurvefordifferentkvalues_

Maximum Precision:- 0. 2302337568649409 at K = 4


_Precisioncurvefordifferentkvalues_

Basedontheaboveplot,optimalkvalueis 3 ,withmaximumf 1 scoreof 0. 33.

Train set accuracy 0. 9294924634950542
Test set accuracy 0. 8798869524258125
[[ 7173 312 ]
[ 708 299 ]]
precision recall f 1 - score support

```
0 0. 91 0. 96 0. 93 7485
1 0. 49 0. 30 0. 37 1007
```
accuracy 0. 88 8492
macro avg 0. 70 0. 63 0. 65 8492
weighted avg 0. 86 0. 88 0. 87 8492

AUCCurve:Figure 1
PrecisionRecallCurve:Figure 2

## ROC-AUCCurve


_Figure 1 :AUCROCCurveforallModels_

## PrecisionRecallCurve

InimbalanceproblemsincewehaveahighnumberofNegatives,thismakesthe
FalsePosiitveRateaslow,resultingintheshiftofROCAUCCurvetowardsleft,
whichisslightlymisleading.

Soinimbalanceproblemweusuallymakesuretolookatprecisionrecallcurveas
well.

_Figure 2 :PrecisionRecallCurveforallModels_

AspertheROCCurveandPrecisionRecallcurve,KNNisperformingbest.Butafter
combiningtheseresultswithprecisionrecallcurve,wesuggestusingRandom
Forestforourproblem.

## Summary

_Table 1 :SummaryofModels_

Model Accuracy Precision Recall AUC

Logistic(Cutoff= 0. 25 ) 0. 88 0. 51 0. 58 0. 872

Logistic(Balanced-Train) 0. 85 0. 49 0. 54

DecisionTree 0. 91 0. 66 0. 47 0. 923


Model Accuracy Precision Recall AUC

RandomForest 0. 88 0. 66 0. 46 0. 913

SVC 0. 89 0. 75 0. 15

LinearSVC 0. 89 0. 62 0. 16

GaussianBayes 0. 88 0. 50 0. 25 0. 841

KNN 0. 92 0. 78 0. 54 0. 965

NaiveBayes 0. 85 0. 56 0. 02

NaiveBayes(Balanced-Train) 0. 69 0. 19 0. 35

SeeTable 1.

## Conclusion

Ourmodelwouldbebeneficialinthefollowingways:

- Fortargetmarketingforbankcampaigns,orinotherevents.Forexample
    basedonthecustomer’sjob,ageandloanhistorythemodelwouldcaneasily
    predictwhetherthecustomerisgoingtosubscribetothetermdepositornot.
    Sooutofthemillionpeople,wecaneasilyshortlistpeoplebasedonour
    modelandspendthetimeonthemsoastoimproveefficiency.
- Improvingbuissnesseffficiencyofbanks.Sinceusingtheedaormodelwe
    caneasilycheckthesubscriptioninsights,itwouldbeveryhelpfulforbanks
    toimprovetheirstratergies.Forexample,basedonthemonthlysubscription
    rates,ifbanksaredecidingthecampaignpromotiontime,itcanimprove
    thereefficiency.
- Since,wehavemonthasainputfactorinourmodel,andallothervaluesare
    static,wecanevenfindthebestmonthtocontactcustomerbasedonthe
    predictedprobabilityofthecustomer.Astherecanbearelationbetweenthe
    jobtypeandthemonththeyaresubscribingortheirfluctuatingbalanceand
    age.Thiscanbeveryusefulinfindingthebesttimetocontact.
- Basedonthemodel,sincethenumberofcontactisplayingamajorrole,ifwe
    havetheoptimaltimetocontactthem,wecanrestrictourcallstolessthan 5
    andfindabetterturnover.
- Wedidn’tseeanyrelationwiththesocialandeconomicfactorshere,butif
    wehadthedataformultipleyears,therewasapossibilityoffindinga
    relation.Ourmodelcanaccomodatethesefactorsaswell,andiftrainedby
    accomodatingthesefactorsaswell,thiscanbehelpfulforbanksinfinding
    thepropertimefortherecampaign.


Hence,analyzingthiskindofmarketingdatasethasgivenusvaluableinsightinto
howwecantweakourmodeltogivebuisnessinsightsaswellascustomerinsights
toimprovesubscriptionoftermdeposits.

## Reference

- https://www.kaggle.com/janiobachmann/bank-marketing-dataset
- (PDF)DataAnalysisofaPortuguesemarketingcampaignusingbank...(no
    date).Availableat:
    https://www.researchgate.net/publication/ 339988208 _Data_Analysis_of_a_
    Portuguese_Marketing_Campaign_using_Bank_Marketing_data_Set(Accessed:
    December 20 , 2022 ).
- Bankmarketingdataset.(n.d.). 1010 data.com.RetrievedDecember 20 , 2022 ,
    from
    https://docs. 1010 data.com/Tutorials/MachineLearningExamples/BankMar
    ketingDataSet_ 2 .html
- Manda,H.,Srinivasan,S.,&Rangarao,D.( 2021 ).IBMCloudPakforData:An
    enterpriseplatformtooperationalizedata,analytics,andAI.PacktPublishing.
- SolvingBankMarketingCalssificationProblem-Databricks.(n.d.).
    Databricks.com.RetrievedDecember 20 , 2022 ,fromhttps://databricks-
    prod-
    cloudfront.cloud.databricks.com/public/ 4027 ec 902 e 239 c 93 eaaa 8714 f 173 b
    cfc/ 8143187682226564 / 2297613386094950 / 3186001515933643 /latest.h
    tml
- SolvingBankMarketingCalssificationProblem-Databricks.(n.d.).
    Databricks.com.RetrievedDecember 20 , 2022 ,fromhttps://databricks-
    prod-
    cloudfront.cloud.databricks.com/public/ 4027 ec 902 e 239 c 93 eaaa 8714 f 173 b
    cfc/ 8143187682226564 / 2297613386094950 / 3186001515933643 /latest.h
    tml
- BankMarketingDataSet.(n.d.).UCIMachineLearningRepository.Retrieved
    December 20 , 2022 ,from
    https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
- https://tradingeconomics.com/


