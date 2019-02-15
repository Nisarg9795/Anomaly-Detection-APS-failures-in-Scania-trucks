
# coding: utf-8

# In[13]:


import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import os
import csv


# In[2]:


def setProportion(features, labels, samples):
    pos_indexes = labels[labels==1].index
    neg_indexes = labels[labels==0].sample(n=samples, replace=False, random_state=0).index
    combine = np.concatenate((pos_indexes,neg_indexes))
    balanced_features = features.loc[combine]
    balanced_labels = labels.loc[combine]
    return balanced_features, balanced_labels


# In[3]:


#Normalization
def preprocess(data):
    scaler = MinMaxScaler()
    scaled_features = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return scaled_features, scaler


# In[4]:


def selectKBest(features, labels, parameters):
    feat_selector = SelectKBest(parameters["score_func"], parameters["kbest"])
    feat_selector.fit_transform(features, labels)
    #Getting indexes of the most important features
    imp_columns =  feat_selector.get_support(indices=True)
    best_features = features.iloc[:, imp_columns]
    return best_features, feat_selector


# In[5]:


def reduceDimensions(data):
    pca = PCA(0.9)
    reduced_features = pca.fit_transform(data)
    reduced_features = pd.DataFrame(reduced_features)
    return reduced_features, pca


# In[6]:


train_data = pd.read_csv("aps_failure_training_set.csv", na_values='na')
test_data = pd.read_csv("aps_failure_test_set.csv", na_values='na')
print(train_data.head())
print(test_data.head())
    #Filling missing values with the median
train_data.fillna(train_data.median(), inplace=True)
test_data.fillna(train_data.median(), inplace=True)
    #Preprocessing datasets
train_labels = train_data['class']
test_labels = test_data['class']
train_features = train_data.drop(columns='class')
test_features = test_data.drop(columns='class')
    #Convert class labels to numeric value for the models to train and test on
train_labels.replace({'neg':0, 'pos':1}, inplace=True)
test_labels.replace({'neg':0, 'pos':1}, inplace=True)


# In[21]:


def main(model, parameters,train_data, test_data, train_labels, test_labels):
    if(model == 'RF'):
        best_train_features, feat_selector  = selectKBest(train_features, train_labels, parameters)
        balanced_features, balanced_labels = setProportion(best_train_features, train_labels, parameters["samples"])
        rfc = RandomForestClassifier(random_state=0)
#         print(balanced_features.shape)
        rfc.fit(balanced_features, balanced_labels)
        best_test_features = feat_selector.transform(test_features)
        #Executa grid search com cross validation
        prediction = rfc.predict(best_test_features)
        #report = classification_report(test_labels, prediction)
        #print(report)
        with open('RF.csv', 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(zip(prediction,test_labels))
        

        cm = confusion_matrix(test_labels, prediction).ravel()
        print("Confusion matrix:")
        print(cm)
        score = cm[1]*10 + cm[2]*500
        print("Score:",score)
        
        return score
    elif(model == "SVM"):
        balanced_features, balanced_labels = setProportion(train_features, train_labels, parameters["samples"])
        scaled_train_features, scalar = preprocess(balanced_features)
        scaled_test_features = scalar.transform(test_features)
        reduced_train_features, pca = reduceDimensions(scaled_train_features)
        svc = svm.SVC()
        svc.fit(reduced_train_features, balanced_labels)
        reduced_test_features = pca.transform(scaled_test_features)
        prediction = svc.predict(reduced_test_features)
        cm = confusion_matrix(test_labels, prediction).ravel()
        print("Confusion matrix:")
        print(cm)
        score = cm[1]*10 + cm[2]*500
        print("Score:",score)
        
        return score
#         print(cm)


# In[8]:


# parameters = {'kbest':150, 'score_func':chi2, 'samples': 1000}
# main('RF', parameters,train_data, test_data, train_labels, test_labels)
# output = []
# for parameter_number in range(170,1,-1):
#     print("-----")
#     print("Number of parameters:",parameter_number)
#     parameters = {'kbest':parameter_number, 'score_func':chi2, 'samples': 1000}
#     score=main('RF', parameters,train_data, test_data, train_labels, test_labels)
#     output.append((parameter_number,score))
# print(output)
    
    


# In[9]:


# # parameters = {'kbest':84, 'score_func':chi2, 'samples': 1000}
# output = []
# for sample_size in range(5000,500,-100):
#     parameters = {'kbest':84, 'score_func':chi2, 'samples': sample_size}
#     score = main('SVM', parameters,train_data, test_data, train_labels, test_labels)
#     output.append((sample_size,score))
# print(output)


# In[23]:


parameters = {'kbest':110, 'score_func':chi2, 'samples': 1000}
main('RF', parameters,train_data, test_data, train_labels, test_labels)

