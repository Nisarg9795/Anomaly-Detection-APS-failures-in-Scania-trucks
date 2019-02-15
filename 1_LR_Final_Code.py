__author__ = 'Sushant'

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# 1. DATA INPUT
# TRAIN DATA
data = pd.read_csv(r"C:\Users\Sushant\Desktop\ASU Courses\CSE575 - Statistical Machine Learning\Final Project/"
                   r"/\3. Implementation\aps_failure_training_set.csv", skiprows=20)
print("Data Read", data.shape)
data = data.replace(['na'], [np.NaN])
X = data.drop(['class'], axis=1)
temp = X.convert_objects(convert_numeric=True)
# temp.fillna(value=0, inplace=True)
temp = temp.fillna(temp.mean()).dropna(axis=1, how='all')
Y = data['class']


# TEST DATA
test_data = pd.read_csv(r"C:\Users\Sushant\Desktop\ASU Courses\CSE575 - Statistical Machine Learning\Final Project/"
                        r"/\3. Implementation\aps_failure_test_set.csv", skiprows=20)
test_data = test_data.replace(['na'], [np.NaN])
X_test = test_data.drop(['class'], axis=1)
X_test = X_test.convert_objects(convert_numeric=True)
# X_test.fillna(value=0, inplace=True)
# X_test = X_test.fillna(X_test.median().dropna(axis=1, how='all')
X_test = X_test.fillna(X_test.mean()).dropna(axis=1, how='all')

Y_test = test_data['class']
print("TEST DATA: ", test_data.shape)

# 2. LR PREDICTION - RAW
'''
# TRAINING
print("Training Started\n")
LogReg = LogisticRegression()
print(LogReg.fit(temp, Y))
print("Training Complete\n")

# ASSESSMENT
print(LogReg.score(X_test, Y_test))
Y_pred = LogReg.predict(X_test)
confusion_matrix = confusion_matrix(Y_test, Y_pred)
print(confusion_matrix)
'''

# 3. NORMILIZATION
scaler = StandardScaler()
scaler.fit(temp)
temp = scaler.transform(temp)
temp = pd.DataFrame(temp)

scaler = StandardScaler()
scaler.fit(X_test)
X_test = scaler.transform(X_test)
X_test = pd.DataFrame(X_test)

# 4. PCA
'''
pca = PCA(0.95)
pca.fit(temp)
N = pca.n_components_
print("PCA: ", N)
temp = pd.DataFrame(pca.transform(temp))

pca = PCA(N)
pca.fit(X_test)
print("TEST_PCA: ", pca.n_components_)
X_test = pd.DataFrame(pca.transform(X_test))
'''
'''
# TRAINING
print("Training Started\n")
LogReg = LogisticRegression()
print(LogReg.fit(temp, Y))
print("Training Complete\n")

# ASSESSMENT
print(LogReg.score(X_test, Y_test))
Y_pred = LogReg.predict(X_test)
confusion_matrix = confusion_matrix(Y_test, Y_pred)
print(confusion_matrix)
'''

# 5. VALIDATION, TRAINING DATA
'''
X_train, X_validation, Y_train, Y_validation = train_test_split(temp, Y, test_size=0.2, random_state=0)
DF = pd.concat([X_train, Y_train], axis=1)
Y_train = Y_train.rename(columns={'class': 'Flag'})

print("Number of data samples in Training: ", len(Y_train))
print("Number of data samples in Validation: ", len(Y_validation))
'''


# 6. UNDERSAMPLING
'''
numberofrecords_pos = len(DF[DF['class'] == 'pos'])
pos_indices = np.array(DF[DF['class'] == 'pos'].index)

neg_indices = np.array(DF[DF['class'] == 'neg'].index)
print(len(pos_indices), len(neg_indices))

random_neg_indices = np.random.choice(neg_indices, numberofrecords_pos, replace=False)
random_neg_indices = np.array(random_neg_indices)
under_sample_indices = np.concatenate([pos_indices, random_neg_indices])
under_sample_data = DF.loc[under_sample_indices, :]
X_undersample = under_sample_data.loc[:, under_sample_data.columns != 'class']
Y_undersample = under_sample_data.loc[:, under_sample_data.columns == 'class']

print("Percentage Neg: ", len(under_sample_data[under_sample_data['class'] == 'neg']) / len(under_sample_data))
print("Percentage Pos : ", len(under_sample_data[under_sample_data['class'] == 'neg']) / len(under_sample_data))
print("Total number of data points : ", len(under_sample_data))
lr = LogisticRegression(C=0.001, penalty='l2')
lr.fit(X_undersample, Y_undersample.values.ravel())

# ASSESSMENT
print(lr.score(X_test, Y_test))
Y_pred = lr.predict(X_test)
confusion_matrix = confusion_matrix(Y_test, Y_pred)
print(confusion_matrix)
'''

# 7. SMOTE
os = SMOTE(random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(temp, Y, test_size=0.3, random_state=0)
columns = X_train.columns

os_data_X, os_data_y = os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
os_data_y = pd.DataFrame(data=os_data_y, columns=['class'])

# CHECK SAMPLES SIZES
print("OverSampled Data Size: ", len(os_data_X))
print("Pos Count: ", len(os_data_y[os_data_y['class'] == 'pos']))
print("Negative in Oversampled data set: ", len(os_data_y[os_data_y['class'] == 'neg']))

print("Ratio of positive in oversampled data: ", len(os_data_y[os_data_y['class'] == 'pos']) / len(os_data_X))
print("Ratio of negative in oversampled data: ", len(os_data_y[os_data_y['class'] == 'neg']) / len(os_data_X))
print(os_data_X.shape, os_data_y.shape)

# TRAINING
'''
print("Training Started\n")
LogReg = LogisticRegression(solver='liblinear')
print(LogReg.fit(os_data_X, os_data_y.values.ravel()))
print("Training Complete\n")

# ASSESSMENT
print(LogReg.score(X_test, Y_test))
Y_pred = LogReg.predict(X_test)
confusion_matrix = confusion_matrix(Y_test, Y_pred)
print(confusion_matrix)
'''

# 8. RECURSIVE FEATURE ELIMINATION
'''
LogReg = LogisticRegression(max_iter=1000) # solver='liblinear')
rfe = RFE(LogReg, 82)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)
Twer = rfe.support_


# Twer = [ True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, False, True, False, True, False, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, False, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, True, True, True, True, True, True, True, True, True, True, True,
#          True, False]


Index = [count for count, elem in enumerate(Twer) if elem == True]
print(Index)
os_data_X = os_data_X[Index]
X_test = X_test[Index]
'''

# 9. PCA
pca = PCA(0.95)
pca.fit(os_data_X)
N = pca.n_components_
print("PCA: ", N)
os_data_X = pd.DataFrame(pca.transform(os_data_X))

pca = PCA(N)
pca.fit(X_test)
print("TEST_PCA: ", pca.n_components_)
X_test = pd.DataFrame(pca.transform(X_test))


# TRAINING
print("Training Started\n")
LogReg = LogisticRegression(solver='liblinear', max_iter=1000)
print(LogReg.fit(os_data_X, os_data_y.values.ravel()))
print("Training Complete\n")

# ASSESSMENT
print(LogReg.score(X_test, Y_test))
Y_pred = LogReg.predict(X_test)
confusion_matrix = confusion_matrix(Y_test, Y_pred)
print(confusion_matrix)

# STORING RESULTS TO A FILE
r = pd.concat([Y_test, pd.DataFrame(Y_pred)], axis=1)
r = r.rename(columns={'class': 'Y_test', 0: 'Y_pred'})
r.to_csv('C:\\Users\Sushant\Desktop\Result.csv')

