import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
import os
import matplotlib.pyplot as plt

np.random.seed(7)

def setProportion(features, labels, samples):
	pos_indexes = labels[labels==1].index
	neg_indexes = labels[labels==0].sample(n=samples, replace=False, random_state=0).index
	balanced_pos_features = features.loc[pos_indexes]
	balanced_neg_features = features.loc[neg_indexes]
	#balanced_neg_features = features.loc[samples]
	return balanced_pos_features, balanced_neg_features

def selectKBest(features, labels, parameters):
    feat_selector = SelectKBest(parameters["score_func"], parameters["kbest"])
    feat_selector.fit_transform(features, labels)
    #Getting indexes of the most important features
    imp_columns =  feat_selector.get_support(indices=True)
    best_features = features.iloc[:, imp_columns]
    return best_features, feat_selector

def main(parameters):
    #Load datasets
    train_data = pd.read_csv("../input/aps_failure_training_set.csv", na_values='na')
    test_data = pd.read_csv("../input/aps_failure_test_set.csv", na_values='na')
    train_data.fillna(train_data.median(), inplace=True)
    test_data.fillna(train_data.median(), inplace=True)
    #Preprocessing datasets
    train_labels = train_data['class']
    test_labels = test_data['class']
    train_features = train_data.drop(columns='class')
    test_features = test_data.drop(columns='class')
    #train_features = preprocess(train_features)
    #test_features = preprocess(test_features)
    train_labels.replace({'neg':0, 'pos':1}, inplace=True)
    test_labels.replace({'neg':0, 'pos':1}, inplace=True)
    #Generating balanced training data
    best_train_features, feat_selector  = selectKBest(train_features, train_labels, parameters)
    balanced_train_pos_features, balanced_train_neg_features = setProportion(best_train_features, train_labels, parameters["neg_sample_indexes"])
    feat_selector.transform(test_features)
    imp_columns =  feat_selector.get_support(indices=True)
    best_test_features = test_features.iloc[:, imp_columns]
    balanced_test_pos_features = best_test_features.loc[test_labels[test_labels==1].index]
    balanced_test_neg_features = best_test_features.loc[test_labels[test_labels==0].index]
    train_mean = balanced_train_neg_features.mean()
    train_deviation = balanced_train_neg_features.std()
    #Important features
    valuable_cols = balanced_train_neg_features.columns
    pos_samples_prob = {}
    #Training Model
    for col in valuable_cols:
        pos_samples_prob[col] = []
    for index in range(len(valuable_cols)):
        sample = balanced_train_pos_features.iloc[index]
        for i, col in enumerate(valuable_cols):
            exp_inside = ((sample[col] - train_mean[i])**2)/(2*(train_deviation[i]**2))
            prob = ((1/(2*np.pi*train_deviation[i])**0.5)*np.exp(-1*exp_inside))
            pos_samples_prob[col].append(prob)
    pos_samples_prob_df = pd.DataFrame.from_dict(pos_samples_prob)
    pos_samples_prob_df_max = pos_samples_prob_df.max()
    
    #Testing Model
    epsilon1 = 0.0001
    epsilon2 = 10
    pos_accuracy = 0
    for index in range(len(balanced_test_pos_features)):
        count = 0
        sample = balanced_test_pos_features.iloc[index]
        for i, col in enumerate(valuable_cols):
            exp_inside = ((sample[col] - train_mean[i])**2)/(2*(train_deviation[i]**2))
            prob = ((1/(2*np.pi*train_deviation[i])**0.5)*np.exp(-1*exp_inside))
            if(prob <= (pos_samples_prob_df_max[col] - epsilon1)):
                count = count + 1
        if(count>=epsilon2):
            pos_accuracy = pos_accuracy + 1
    #print(pos_accuracy)
    
    pos_inaccuracy = 0
    for index in range(len(balanced_test_neg_features)):
        count = 0
        sample = balanced_test_neg_features.iloc[index]
        for i, col in enumerate(valuable_cols):
            exp_inside = ((sample[col] - train_mean[i])**2)/(2*(train_deviation[i]**2))
            prob = ((1/(2*np.pi*train_deviation[i])**0.5)*np.exp(-1*exp_inside))
            if(prob <= (pos_samples_prob_df_max[col] - epsilon1)):
                count = count + 1
        if(count>=epsilon2):
            pos_inaccuracy = pos_inaccuracy + 1
    #print(pos_inaccuracy)
    
    return pos_accuracy, pos_inaccuracy
    
if __name__ == "__main__":
    cost_list = []
    iterations = []
    pos = []
    neg = []
    Kbest = [35] #58, 40, 35, 35, 35
    sample_indexes = [800] #1000, 1000, 1000, 900, 800
    for i in range(1):
        parameters = {'kbest':Kbest[i], 'score_func':chi2, 'neg_sample_indexes': sample_indexes[i]}
        pos_accuracy,  pos_inaccuracy = main(parameters)
        pos.append(pos_accuracy)
        neg.append(pos_inaccuracy)
        cost = (375 - pos_accuracy)*500 + pos_inaccuracy*10
        cost_list.append(cost)
        iterations.append(i+1)
    plt.plot(iterations, cost_list)
    plt.ylabel('Cost')
    plt.xlabel('Iteration')
    plt.show()
    print(pos, neg)
    '''
    neg_indexes = np.arange(60000)
    pos_accuracy = []
    pos_inaccuracy = []
    for i in range(60):
        np.random.shuffle(neg_indexes)
        neg_sample_indexes = neg_indexes[:1000]
        parameters = {'kbest':84, 'score_func':chi2, 'neg_sample_indexes': neg_sample_indexes}
        neg_indexes = np.setdiff1d(neg_indexes, neg_sample_indexes)
        main(parameters)'''
