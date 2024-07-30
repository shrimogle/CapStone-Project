#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[2]:


df = pd.read_csv('data.csv',sep='\t')
df.info()


# # Data Cleaning

# In[3]:


df['wrongansw'] = 0
df['wrongansw'] = df['wrongansw'].where(df['VCL6']== 0, df['wrongansw'] + 1)
df['wrongansw'] = df['wrongansw'].where(df['VCL9']== 0, df['wrongansw'] + 1)
df['wrongansw'] = df['wrongansw'].where(df['VCL12']== 0, df['wrongansw'] + 1)

df.wrongansw.value_counts()


# In[4]:


df = df[df['wrongansw'].isin([2, 3])]
df = df.drop(columns='wrongansw')
df.head(3)
df.shape


# In[5]:


vcls = []
for i in range(1, 17):
    vcls.append('VCL' + str(i))
    
df = df.drop(columns=vcls)
df.shape


# In[6]:


categorical = df.select_dtypes('object').columns

print('Categorical Columns: ', df[categorical].columns)
print(df[categorical].nunique())

df = df.drop(['major'], axis=1)


# # Labelling the questions that are for specific mental disorder

# In[7]:


DASS_keys = {
    'Depression': [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42],
    'Anxiety': [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41],
    'Stress': [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39]
    }


# # Data Collection

# In[8]:


depr = []
for i in DASS_keys["Depression"]:
    depr.append('Q' + str(i) + 'A')
    
anx = []
for i in DASS_keys["Anxiety"]:
    anx.append('Q' + str(i) + 'A')

stre = []
for i in DASS_keys["Stress"]:
    stre.append('Q' + str(i) + 'A')


df_depr=df.filter(depr) 
df_anx=df.filter(anx)
df_stre=df.filter(stre)


# In[9]:


disorders = [depr, anx, stre]

def scores(df):        
    df["ScoresDepr"] = df[depr].sum(axis=1) 
    df["ScoresAnx"] = df[anx].sum(axis=1)
    df["ScoresStre"] = df[stre].sum(axis=1)
    return df

for i in disorders:
        df[i] -= 1 
        
df = scores(df)
        
df.head()


# In[10]:


# CATEGORY: DEPRESSION
CategoryDepr=[]

for i in df['ScoresDepr']:
    if i in range(0,10):
        CategoryDepr.append('Normal')
    elif i in range(10,14):
        CategoryDepr.append('Minimal')
    elif i in range(14,21):
        CategoryDepr.append('Moderate')
    elif i in range(21,28):
        CategoryDepr.append('Severe')
    else:
        CategoryDepr.append('Extremely Severe')
        
df['CategoryDepr']= CategoryDepr


# In[11]:


# CATEGORY: ANXIETY
CategoryAnx=[]

for i in df['ScoresAnx']:
    if i in range(0,8):
        CategoryAnx.append('Normal')
    elif i in range(8,10):
        CategoryAnx.append('Minimal')
    elif i in range(10,15):
        CategoryAnx.append('Moderate')
    elif i in range(15,20):
        CategoryAnx.append('Severe')
    else:
        CategoryAnx.append('Extremely Severe')
        
df['CategoryAnx']= CategoryAnx


# In[12]:


# CATEGORY: STRESS
CategoryStre=[]

for i in df['ScoresStre']:
    if i in range(0,15):
        CategoryStre.append('Normal')
    elif i in range(15,19):
        CategoryStre.append('Minimal')
    elif i in range(19,26):
        CategoryStre.append('Moderate')
    elif i in range(26,34):
        CategoryStre.append('Severe')
    else:
        CategoryStre.append('Extremely Severe')
        
df['CategoryStre']= CategoryStre


# In[13]:


df.isnull().sum()
df.duplicated().sum()


# In[14]:


df


# # Data Classification Models for Depression

#  ### Data Pre-processing

# In[15]:


Y = df['CategoryDepr']
X = df.drop(columns=['Q2A','Q4A','Q7A','Q9A','Q15A','Q19A','Q20A','Q23A','Q25A','Q28A','Q30A','Q36A','Q40A','Q41A','introelapse','testelapse', 'surveyelapse','engnat','CategoryAnx', 'CategoryDepr', 'CategoryStre','country', 'ScoresAnx' ,'ScoresStre','screensize','uniquenetworklocation','Q1A' ,'Q1I', 'Q1E', 'Q2I', 'Q2E', 'Q3A', 'Q3I', 'Q3E','Q4I','Q4E', 'Q5E', 'Q5A' ,'Q5I', 'Q6E', 'Q6A' ,'Q6I', 'Q7E' ,'Q7I', 'Q8E','Q8A' ,'Q8I', 'Q9E', 'Q9I', 'Q10E', 'Q10A', 'Q10I', 'Q11E', 'Q11A' ,'Q11I', 'Q12E', 'Q12A' ,'Q12I', 'Q13E', 'Q13A' ,'Q13I', 'Q14E', 'Q14A' ,'Q14I', 'Q15E', 'Q15I', 'Q16E', 'Q16A' ,'Q16I', 'Q17E', 'Q17A' ,'Q17I', 'Q18E', 'Q18A' ,'Q18I', 'Q19I', 'Q19E',  'Q20I', 'Q20E', 'Q21A', 'Q21I', 'Q21E', 'Q22A', 'Q22I','Q22E', 'Q23I','Q23E','Q24A','Q24I','Q24E', 'Q25I', 'Q25E', 'Q26A', 'Q26I', 'Q26E', 'Q27A', 'Q27I', 'Q27E', 'Q28I', 'Q28E', 'Q29A', 'Q29I', 'Q29E',  'Q30I', 'Q30E', 'Q31A', 'Q31I', 'Q31E', 'Q32A', 'Q32I', 'Q32E', 'Q33A', 'Q33I', 'Q33E', 'Q34A', 'Q34I', 'Q34E', 'Q35A', 'Q35I', 'Q35E', 'Q36I', 'Q36E', 'Q37A', 'Q37I', 'Q37E', 'Q38A', 'Q38I', 'Q38E', 'Q39A', 'Q39I', 'Q39E', 'Q40I', 'Q40E', 'Q41I', 'Q41E', 'Q42A', 'Q42I', 'Q42E'])
X.head()


# In[16]:


scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


# In[17]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=85)


# # Classification Models For Depression

# In[18]:


# KNN Model for Depression Classification

knn = KNeighborsClassifier(n_neighbors=95)


knn.fit(X_train, Y_train)


Y_pred = knn.predict(X_test)


accuracy_knn = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
classif_report = classification_report(Y_test, Y_pred)

print('Accuracy of KNN Model - Depression: {:.5f}'.format(accuracy_knn))
print('Confusion Matrix of KNN Model - Depression: \n', conf_matrix)
print('Confusion Matrix of KNN Model - Depression: \n', classif_report) 


# In[19]:


# SVC Model for Depression Classification
svm_model = SVC(kernel='linear')

svm_model.fit(X_train, Y_train)

preds = svm_model.predict(X_test)


accuracy_svc = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
classif_report = classification_report(Y_test, Y_pred)

print('Accuracy of SVM Model - Depression: {:.5f}'.format(accuracy_svc))
print('Confusion Matrix of SVM Model - Depression: \n', conf_matrix)
print('Confusion Matrix of SVM Model - Depression: \n', classif_report) 


# In[20]:


# Naive Bayes Model for Depression Classification

clfNB = GaussianNB()


clfNB.fit(X_train, Y_train)


Y_pred = clfNB.predict(X_test)


accuracy_nb = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
classif_report = classification_report(Y_test, Y_pred)

print('Accuracy of NB Model - Depression: {:.5f}'.format(accuracy_nb))
print('Confusion Matrix of NB Model - Depression: \n', conf_matrix)
print('Classification Report of NB Model - Depression: \n', classif_report) 


# In[21]:


# Random Forest Classifier Model for Depression Classification

clfRFC = RandomForestClassifier(n_estimators=110, random_state=110)


clfRFC.fit(X_train, Y_train)


Y_pred = clfRFC.predict(X_test)


accuracy_rfc = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
classif_report = classification_report(Y_test, Y_pred)


print('Accuracy of RFC Model - Depression: {:.5f}'.format(accuracy_rfc))
print('Confusion Matrix of RFC Model - Depression: \n', conf_matrix)
print('Classification report of RFC Model - Depression: \n', classif_report)  


# In[22]:


accuracy_depression = [accuracy_knn, accuracy_svc, accuracy_nb, accuracy_rfc ]

depression_analysis = pd.DataFrame({
                        'Models' : ['KNN', 'SVC', 'Naive Bayes', 'Random Forest'],
                        'Accuracy_Depression': [x * 100 for x in accuracy_depression]
                    })


# # Data Classification Models for Anxiety

# ### Data Pre-processing

# In[23]:


Y = df['CategoryAnx']
X = df.drop(columns=['Q2A','Q4A','Q7A','Q9A','Q15A','Q19A','Q20A','Q23A','Q25A','Q28A','Q30A','Q36A','Q40A','Q41A','introelapse','testelapse', 'surveyelapse','engnat','CategoryAnx','CategoryDepr', 'CategoryStre','country', 'ScoresDepr' ,'ScoresStre','screensize','uniquenetworklocation','Q1A' ,'Q1I', 'Q1E', 'Q2I', 'Q2E', 'Q3A', 'Q3I', 'Q3E','Q4I','Q4E', 'Q5E', 'Q5A' ,'Q5I', 'Q6E', 'Q6A' ,'Q6I', 'Q7E' ,'Q7I', 'Q8E','Q8A' ,'Q8I', 'Q9E', 'Q9I', 'Q10E', 'Q10A', 'Q10I', 'Q11E', 'Q11A' ,'Q11I', 'Q12E', 'Q12A' ,'Q12I', 'Q13E', 'Q13A' ,'Q13I', 'Q14E', 'Q14A' ,'Q14I', 'Q15E', 'Q15I', 'Q16E', 'Q16A' ,'Q16I', 'Q17E', 'Q17A' ,'Q17I', 'Q18E', 'Q18A' ,'Q18I', 'Q19I', 'Q19E',  'Q20I', 'Q20E', 'Q21A', 'Q21I', 'Q21E', 'Q22A', 'Q22I','Q22E', 'Q23I','Q23E','Q24A','Q24I','Q24E', 'Q25I', 'Q25E', 'Q26A', 'Q26I', 'Q26E', 'Q27A', 'Q27I', 'Q27E', 'Q28I', 'Q28E', 'Q29A', 'Q29I', 'Q29E',  'Q30I', 'Q30E', 'Q31A', 'Q31I', 'Q31E', 'Q32A', 'Q32I', 'Q32E', 'Q33A', 'Q33I', 'Q33E', 'Q34A', 'Q34I', 'Q34E', 'Q35A', 'Q35I', 'Q35E', 'Q36I', 'Q36E', 'Q37A', 'Q37I', 'Q37E', 'Q38A', 'Q38I', 'Q38E', 'Q39A', 'Q39I', 'Q39E', 'Q40I', 'Q40E', 'Q41I', 'Q41E', 'Q42A', 'Q42I', 'Q42E'])
X.head()


# In[24]:


scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


# In[25]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=85)


# # Classification Models For Anxiety

# In[26]:


# KNN Classification

knn = KNeighborsClassifier(n_neighbors=95)


knn.fit(X_train, Y_train)


Y_pred = knn.predict(X_test)


accuracy_knn = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
classif_report = classification_report(Y_test, Y_pred)

print('Accuracy of KNN Model - Anxiety: {:.5f}'.format(accuracy_knn))
print('Confusion Matrix of KNN Model - Anxiety: \n', conf_matrix)
print('Confusion Matrix of KNN Model - Anxiety: \n', classif_report)


# In[27]:


# SVC Model for Anxiety Classification
svm_model = SVC(kernel='linear')

svm_model.fit(X_train, Y_train)

preds = svm_model.predict(X_test)


accuracy_svc = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
classif_report = classification_report(Y_test, Y_pred)

print('Accuracy of SVM Model - Anxiety: {:.5f}'.format(accuracy_svc))
print('Confusion Matrix of SVM Model - Anxiety: \n', conf_matrix)
print('Confusion Matrix of SVM Model - Anxiety: \n', classif_report)


# In[28]:


# Naive Bayes Model for Anxiety Classification
clfNB = GaussianNB()


clfNB.fit(X_train, Y_train)


Y_pred = clfNB.predict(X_test)


accuracy_nb = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
classif_report = classification_report(Y_test, Y_pred)

print('Accuracy of NB Model - Anxiety: {:.5f}'.format(accuracy_nb))
print('Confusion Matrix of NB Model - Anxiety: \n', conf_matrix)
print('Classification Report of NB Model - Anxiety: \n', classif_report)


# In[29]:


# Random Forest Classifier Model for Anxiety Classification
clfRFC = RandomForestClassifier(n_estimators=95, random_state=85)


clfRFC.fit(X_train, Y_train)


Y_pred = clfRFC.predict(X_test)


accuracy_rfc = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
classif_report = classification_report(Y_test, Y_pred)


print('Accuracy of RFC Model - Anxiety: {:.5f}'.format(accuracy_rfc))
print('Confusion Matrix of RFC Model - Anxiety: \n', conf_matrix)
print('Classification report of KNN Model - Anxiety: \n', classif_report)


# In[30]:


accuracy_anxiety = [accuracy_knn, accuracy_svc, accuracy_nb, accuracy_rfc ]

anxiety_analysis = pd.DataFrame({
                        'Models' : ['KNN', 'SVC', 'Naive Bayes', 'Random Forest' ],
                        'Accuracy_Anxiety': [x * 100 for x in accuracy_anxiety]
                    })


# # Data Classification Models for Stress

# ### Data Preprocessing

# In[31]:


Y = df['CategoryStre']
X = df.drop(columns=['Q2A','Q4A','Q7A','Q9A','Q15A','Q19A','Q20A','Q23A','Q25A','Q28A','Q30A','Q36A','Q40A','Q41A','introelapse','testelapse', 'surveyelapse','engnat','CategoryStre','CategoryAnx', 'CategoryDepr', 'country', 'ScoresDepr' ,'ScoresAnx','screensize','uniquenetworklocation','Q1A' ,'Q1I', 'Q1E', 'Q2I', 'Q2E', 'Q3A', 'Q3I', 'Q3E','Q4I','Q4E', 'Q5E', 'Q5A' ,'Q5I', 'Q6E', 'Q6A' ,'Q6I', 'Q7E' ,'Q7I', 'Q8E','Q8A' ,'Q8I', 'Q9E', 'Q9I', 'Q10E', 'Q10A', 'Q10I', 'Q11E', 'Q11A' ,'Q11I', 'Q12E', 'Q12A' ,'Q12I', 'Q13E', 'Q13A' ,'Q13I', 'Q14E', 'Q14A' ,'Q14I', 'Q15E', 'Q15I', 'Q16E', 'Q16A' ,'Q16I', 'Q17E', 'Q17A' ,'Q17I', 'Q18E', 'Q18A' ,'Q18I', 'Q19I', 'Q19E',  'Q20I', 'Q20E', 'Q21A', 'Q21I', 'Q21E', 'Q22A', 'Q22I','Q22E', 'Q23I','Q23E','Q24A','Q24I','Q24E', 'Q25I', 'Q25E', 'Q26A', 'Q26I', 'Q26E', 'Q27A', 'Q27I', 'Q27E', 'Q28I', 'Q28E', 'Q29A', 'Q29I', 'Q29E',  'Q30I', 'Q30E', 'Q31A', 'Q31I', 'Q31E', 'Q32A', 'Q32I', 'Q32E', 'Q33A', 'Q33I', 'Q33E', 'Q34A', 'Q34I', 'Q34E', 'Q35A', 'Q35I', 'Q35E', 'Q36I', 'Q36E', 'Q37A', 'Q37I', 'Q37E', 'Q38A', 'Q38I', 'Q38E', 'Q39A', 'Q39I', 'Q39E', 'Q40I', 'Q40E', 'Q41I', 'Q41E', 'Q42A', 'Q42I', 'Q42E'])
X.head()


# In[32]:


scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


# In[33]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=85)


# # Classification Models For Stress

# In[34]:


# KNN Model for Stress Classification


knn = KNeighborsClassifier(n_neighbors=95)


knn.fit(X_train, Y_train)


Y_pred = knn.predict(X_test)


accuracy_knn = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
classif_report = classification_report(Y_test, Y_pred)

print('Accuracy of KNN Model - Stress: {:.5f}'.format(accuracy_knn))
print('Confusion Matrix of KNN Model - Stress: \n', conf_matrix)
print('Confusion Matrix of KNN Model - Stress: \n', classif_report)


# In[35]:


# SVC Model for Stress Classification
svm_model = SVC(kernel='linear')

svm_model.fit(X_train, Y_train)

preds = svm_model.predict(X_test)


accuracy_svc = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
classif_report = classification_report(Y_test, Y_pred)

print('Accuracy of SVM Model - Anxiety: {:.5f}'.format(accuracy_svc))
print('Confusion Matrix of SVM Model - Anxiety: \n', conf_matrix)
print('Confusion Matrix of SVM Model - Anxiety: \n', classif_report)


# In[36]:


# Naive Bayes Model for Stress Classification

clfNB = GaussianNB()


clfNB.fit(X_train, Y_train)


Y_pred = clfNB.predict(X_test)


accuracy_nb = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
classif_report = classification_report(Y_test, Y_pred)

print('Accuracy of NB Model - Stress: {:.5f}'.format(accuracy_nb))
print('Confusion Matrix of NB Model - Stress: \n', conf_matrix)
print('Classification Report of KNN Model - Stress: \n', classif_report)


# In[37]:


# Random Forest Classifier Model for Stress Classification

clfRFC = RandomForestClassifier(n_estimators=95, random_state=85)


clfRFC.fit(X_train, Y_train)


Y_pred = clfRFC.predict(X_test)


accuracy_rfc = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
classif_report = classification_report(Y_test, Y_pred)


print('Accuracy of RFC Model - Stress: {:.5f}'.format(accuracy_rfc))
print('Confusion Matrix of RFC Model - Stress: \n', conf_matrix)
print('Classification report of KNN Model - Stress: \n', classif_report)


# In[38]:


accuracy_stress = [accuracy_knn, accuracy_svc, accuracy_nb, accuracy_rfc ]

stress_analysis = pd.DataFrame({
                        'Models' : ['KNN', 'SVC', 'Naive Bayes', 'Random Forest' ],
                        'Accuracy_Stress': [x * 100 for x in accuracy_stress]
                    })


# # Analysis of the Models Used

# In[39]:


analysis = pd.concat([stress_analysis.set_index('Models'), 
                      anxiety_analysis.set_index('Models'), 
                      depression_analysis.set_index('Models')],
                     axis=1)


# In[40]:


analysis.head()


# In[41]:


ax = analysis.plot(kind='barh')
plt.title("Accuracy of classification using different ML algorithms for DASS42")
plt.xlabel("Accuracy")
plt.ylabel("Models")
plt.show()

