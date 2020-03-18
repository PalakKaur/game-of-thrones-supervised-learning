#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 01:07:37 2019

@author: palakkaur

Purpose:
    This code is meant for creating and comparing various 
    machine learning models to predict survival rate in 
    Game of Thrones series.
"""

###############################################################################
# Importing libraries and base dataset
###############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve


got_df = pd.read_excel('GOT_character_predictions.xlsx')
got_dictionary = pd.read_excel('GOT_data_dictionary.xlsx')


###############################################################################
# Fundamental Dataset Exploration
###############################################################################

# Column names
got_df.columns

# Displaying first rows of the DataFrame
got_df.head()

# Information about each variable
got_df.info()

# Descriptive statistics
got_df.describe().round(2)


###############################################################################
# Checking for missing values
###############################################################################

print(
      got_df.isnull()
      .any()
      )

# Missing value count in each column
print(
      got_df[:]
      .isnull()
      .sum()
      )

# Percentage of missing values in each column
print(
      ((got_df[:].isnull().sum())
      /
      got_df[:]
      .count()).round(2).sort_values(ascending = False)
     ) 
 

# Creating a copy of df
got_copy = pd.DataFrame.copy(got_df)

     
# Flagging missing values
for col in got_copy:
    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    if got_copy[col].isnull().astype(int).sum() > 0: 
        got_copy['m_'+col] = got_copy[col].isnull().astype(int)      


# Only noble characters have titles
got_copy['title'] = got_copy['title'].fillna('Not Noble')


# Assuming people with missing info are not alive
got_copy['isAliveMother'] = got_copy['isAliveMother'].fillna(0)

got_copy['isAliveHeir'] = got_copy['isAliveHeir'].fillna(0)

got_copy['isAliveFather'] = got_copy['isAliveFather'].fillna(0)

got_copy['isAliveSpouse'] = got_copy['isAliveSpouse'].fillna(0)


# Imputation with median for numeric variables
for col in got_copy:
    """ Impute missing values using the median of each column """
    if (got_copy[col].isnull().astype(int).sum() > 0 and 
        is_numeric_dtype(got_copy[col]) == True):
        col_median = got_copy[col].median()
        got_copy[col] = got_copy[col].fillna(col_median).round(2)
 
    
# Imputation with string for categorical variables to move further w/ analysis 
for col in got_copy:
    """ Impute missing values using a string """
    if (got_copy[col].isnull().astype(int).sum() > 0 and 
        is_string_dtype(got_copy[col]) == True):
        got_copy[col] = got_copy[col].fillna('Unavailable')

# Checking for missing values
print(
      got_copy[:]
      .isnull()
      .sum()
      ) 
        

# Correlation Matrices
df_corr = got_copy.corr()
df_corr.loc['isAlive'].sort_values(ascending = True)


###############################################################################
# Dealing with categorical variables 
###############################################################################

# Title cleaning

got_copy['title'].unique()

got_copy['title'][got_copy['title'].str.contains('King')] = 'King'


got_copy['title'][got_copy['title'].str.contains('Prince')] = 'Prince'


got_copy['title'][got_copy['title'].str.contains('Queen')] = 'Queen'


got_copy['title'][got_copy['title'].str.contains('Lady')] = 'Lady'


got_copy['title'][got_copy['title'].str.contains('aester')] = 'Maester'


got_copy['title'][got_copy['title'].str.contains('Lord')] = 'Lord'


got_copy['title'][got_copy['title'].str.contains('Commander')] = 'Commander'


got_copy['title'][got_copy['title'].str.contains('Knight')] = 'Knight'


got_copy['title'][got_copy['title'].str.contains('Sept')] = 'Septon'



# Culture cleaning

got_copy['culture'].nunique()

got_copy['culture'] = got_copy['culture'].replace(['Vale',
                                                   'Vale mountain clans'], 
                                                   'Valemen')

got_copy['culture'] = got_copy['culture'].replace(["Braavosi"], 'Braavos')

got_copy['culture'] = got_copy['culture'].replace(['Northern mountain clans',
                                                   'northmen'],
                                                   'Northmen')

got_copy['culture'] = got_copy['culture'].replace(['Dornish', 'Dornishmen'], 
                                                         'Dorne')

got_copy['culture'] = got_copy['culture'].replace(['Lyseni'], 'Lysene')

# Free folk are also called wildlings
got_copy['culture'] = got_copy['culture'].replace(['Free folk', 'free folk',
                                                   'Wildling','Wildlings'],
                                                         'Free Folk')

got_copy['culture'] = got_copy['culture'].replace(['ironborn', 'Ironmen'], 
                                                         'Ironborn')

got_copy['culture'] = got_copy['culture'].replace(['Reachmen', 'The Reach'], 
                                                         'Reach')

got_copy['culture'] = got_copy['culture'].replace(['Ghiscaricari'], 
                                                         'Ghiscari')

got_copy['culture'] = got_copy['culture'].replace(['Astapori'], 'Astapor')

got_copy['culture'] = got_copy['culture'].replace(["Qartheen"], 'Qarth')

got_copy['culture'] = got_copy['culture'].replace(['Riverlands'], 'Rivermen')

got_copy['culture'] = got_copy['culture'].replace(['Westerman', 'Westermen',
                                                   'westermen', 'Westerlands'], 
                                                   'Westeros')

got_copy['culture'] = got_copy['culture'].replace(['Meereenese'], 'Meereen')

got_copy['culture'] = got_copy['culture'].replace(['Lhazarene', 'Lhazrene', 
                                                   'Lhazreen'], 'Lhazareen')

got_copy['culture'] = got_copy['culture'].replace(['Stormlander'], 
                                                         'Stormlands')

got_copy['culture'] = got_copy['culture'].replace(["Asshai'i"], 'Asshai')

got_copy['culture'] = got_copy['culture'].replace(['Norvoshi'], 'Norvos')

got_copy['culture'] = got_copy['culture'].replace(['Summer Islander', 
                                                         'Summer Islands'], 
                                                         'Summer Isles')

got_copy['culture'] = got_copy['culture'].replace(['Andals'], 'Andal')

got_copy['culture'] = got_copy['culture'].replace(['Lhazrene', 'Lhazarene', 
                                                         'Lhazreen'],
                                                         'Lhazareen')

got_copy['culture'] = got_copy['culture'].replace(['Norvoshi'], 'Norvos')


got_copy['culture'] = got_copy['culture'].replace(['Stormlander'], 
                                                         'Stormlands')

 # Creating a numeric variable for culture
got_copy['culture_num'] = pd.factorize(got_copy['culture'], sort=True)[0] + 1

# Creating culture dictionary
culture_dict = got_copy.filter(['culture_num', 'culture'])



# House cleaning

got_copy['house'].unique()

got_copy['house'][got_copy['house'].str.contains('Tyrell')] = 'House Tyrell'


got_copy['house'][got_copy['house'].str.contains('Martell')] = 'House Martell'


got_copy['house'][got_copy['house'].str.contains('Royce')] = 'House Royce'


got_copy['house'][got_copy['house'].str.contains('Frey')] = 'House Frey'


(got_copy['house'][got_copy['house']
                    .str.contains('Fossoway')]) = 'House Fossoway'


got_copy['house'][got_copy['house'].str.contains('Brune')] = 'House Brune'


(got_copy['house'][got_copy['house']
                    .str.contains('Baratheon')]) = 'House Baratheon'


(got_copy['house'][got_copy['house']
                    .str.contains('Lannister')]) = 'House Lannister'


(got_copy['house'][got_copy['house']
                    .str.contains('Goodbrother')]) = 'House Goodbrother'


got_copy['house'][got_copy['house'].str.contains('Bolton')] = 'House Bolton'


got_copy['house'][got_copy['house'].str.contains('Harlaw')] = 'House Harlaw'


got_copy['house'][got_copy['house'].str.contains('Kenning')] = 'House Kenning'


got_copy['house'][got_copy['house'].str.contains('Dayne')] = 'House Dayne'


got_copy['house'][got_copy['house'].str.contains('Vance')] = 'House Vance'


got_copy['house'][got_copy['house'].str.contains('Royce')] = 'House Royce'


got_copy['house'][got_copy['house'].str.contains('Tyrell')] = 'House Tyrell'


got_copy['house'][got_copy['house'].str.contains('Farwynd')] = 'House Farwynd'


got_copy['house'][got_copy['house'].str.contains('Frey')] = 'House Frey'


got_copy['house'][got_copy['house'].str.contains('Flint')] = 'House Flint'


# Creating a numeric variable for house
got_copy['house_num'] = pd.to_numeric(pd.factorize(got_copy['house'], 
                                                    sort=True)[0] + 1)

# Creating house dictionary
house_dict = got_copy.filter(['house_num', 'house'])


###############################################################################
#Outlier Detection
###############################################################################
descriptive_stats = got_copy.describe().round(2)

# Boxplots   
for col in got_copy.loc[:, ['dateOfBirth',
                            'age',
                            'popularity',
                            'numDeadRelations',
                            'house_num',
                            'culture_num']]:
        got_copy.boxplot(column = col, vert = False)
        plt.title(f"{col}")
        plt.tight_layout()
        plt.show()

# Histograms
for col in got_copy.iloc[:, 1: ]:
    if is_numeric_dtype(got_copy[col]) == True:
      sns.distplot(got_copy[col], kde = True)
      plt.tight_layout()
      plt.show()


# Setting threshold
dateOfBirth = 260
dateOfBirth_low = -25
age = 35
age_low = 0
numDeadRelations = 3
popularity = 0.2
house_code_l = 95
culture_code_l = 27


###############################################################################
# FLAG OUTLIERS
###############################################################################

# Creating functions to flag upper and lower limits

def up_out(col, lim):
    got_copy['o_'+col] = 0
    for val in enumerate(got_copy.loc[ : , col]):   
        if val[1] > lim:
            got_copy.loc[val[0], 'o_'+col] = 1 
                
def low_out(col, lim):
    got_copy['o_'+col] = 0
    for val in enumerate(got_copy.loc[ : , col]):   
        if val[1] < lim:
            got_copy.loc[val[0], 'o_'+col] = 1      
            
# Flagging upper outliers
up_out('numDeadRelations', numDeadRelations)
up_out('popularity', popularity)
low_out('house_num', house_code_l)
low_out('culture_num', culture_code_l)

# Flagging upper and lower outliers for dob and age
got_copy['o_dateOfBirth'] = 0
for val in enumerate(got_copy.loc[ : , 'dateOfBirth']):    
    if val[1] < dateOfBirth_low:
        got_copy.loc[val[0], 'o_dateOfBirth'] = -1
    elif val[1] > dateOfBirth:
        got_copy.loc[val[0], 'o_dateOfBirth'] = 1

got_copy['o_age'] = 0
for val in enumerate(got_copy.loc[ : , 'age']):    
    if val[1] < age_low:
        got_copy.loc[val[0], 'o_age'] = -1
    elif val[1] > age:
        got_copy.loc[val[0], 'o_age'] = 1



# Creating dummies for categorical data - culture and house
title_dummies = pd.get_dummies((got_copy['title']), 
                                drop_first = True)

culture_dummies = pd.get_dummies((got_copy['culture']), 
                                  drop_first = True)

house_dummies = pd.get_dummies((got_copy['house']), 
                                drop_first = True)

got_withdummies = pd.concat(
                            [got_copy.loc[:, :],
                             title_dummies, culture_dummies, house_dummies],
                             axis = 1)



# Creating feature for main houses based on books

got_withdummies['main_houses'] = 0

got_withdummies['main_houses'] = (
                                  got_withdummies['House Targaryen'] +
                                  got_withdummies['House Lannister'] +
                                  got_withdummies['House Stark'] +
                                  got_withdummies['House Baratheon'] +
                                  got_withdummies['House Tyrell'] +
                                  got_withdummies['House Martell'] +
                                  got_withdummies['House Arryn'] +
                                  got_withdummies['House Greyjoy'] +
                                  got_withdummies['House Tully'])


# Creating feature for houses more likely to stay alive

print(got_withdummies.groupby(["house", "isAlive"]).count()["S.No"].
      unstack().copy(deep = True).sort_values(by = 1, ascending = False))
# Targaryen is not safe

got_withdummies['safe_houses'] = 0

got_withdummies['safe_houses'] = (
                            got_withdummies['House Frey'] +
                            got_withdummies['House Tyrell'] +
                            got_withdummies['House Martell'] +
                            got_withdummies['House Baratheon'] +
                            got_withdummies['Brotherhood Without Banners'] +
                            got_withdummies['House Vance'])


# Creating new feature for royals

got_withdummies['royals'] = 0

got_withdummies['royals'] = (
                             got_withdummies['King'] +
                             got_withdummies['Queen'] +
                             got_withdummies['Prince'] +
                             got_withdummies['Lord'])


# Creating new feature for safe titles

print(got_withdummies.groupby(["title", "isAlive"]).count()["S.No"].
      unstack().copy(deep = True).sort_values(by = 1, ascending = False))

got_withdummies['safe_titles'] = 0

got_withdummies['safe_titles'] = (
                             got_withdummies['Maester'] +
                             got_withdummies['Septon'] +
                             got_withdummies['Ser'])

# Creating new features for safe and unsafe cultures
print(got_withdummies.groupby(["culture", "isAlive"]).count()["S.No"].
      unstack().copy(deep = True).sort_values(by = 0, ascending = False))

got_withdummies['safe_cultures'] = (
                                    got_withdummies['Northmen'] + 
                                    got_withdummies['Ironborn'] +
                                    got_withdummies['Braavos'] +
                                    got_withdummies['Dorne'])
###############################################################################
# Logistic Regression with statistically significant variables
###############################################################################

logistic = smf.logit(formula = """isAlive ~    male + 
                                                    book1_A_Game_Of_Thrones +
                                                    book3_A_Storm_Of_Swords +
                                                    book4_A_Feast_For_Crows +
                                                    o_popularity+
                                                    o_dateOfBirth
                                                     """
                                                    ,data = got_copy)

results_logistic = logistic.fit()

results_logistic.summary()

# Checking AIC and BIC
print('AIC:', results_logistic.aic.round(2))
print('BIC:', results_logistic.bic.round(2))


########################
# Machine Learning using KNN - Testing Score: 0.82
########################

# Preparing data for model
got_data = got_withdummies.loc[: , ['male',
                             'book1_A_Game_Of_Thrones', 
                             'book3_A_Storm_Of_Swords',
                             'book4_A_Feast_For_Crows',
                             'numDeadRelations',
                             'dateOfBirth',
                             'popularity'
                              ]]

got_target =  got_withdummies.loc[: , 'isAlive']


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.1,
            random_state = 508,
            stratify = got_target) # stratify


# List for training and test accuracy
training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 51)

for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train.values.ravel())
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


fig, ax = plt.subplots(figsize = (12,9))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

# Highest test accuracy
print(max(test_accuracy))

# Optimal number of neighbours
print(test_accuracy.index(max(test_accuracy)) + 1)

# Building model with k = 10
knn_clf = KNeighborsClassifier(n_neighbors = 10)

# Fitting the model based on the training data
knn_clf_fit = knn_clf.fit(X_train, y_train)

# Scoring the model
y_score_knn_optimal = knn_clf.score(X_test, y_test)

print(y_score_knn_optimal)

# Generating Predictions based on the optimal KNN model
knn_clf_optimal_pred = knn_clf_fit.predict(X_test)


print('Training Score', knn_clf_fit.score(X_train, y_train).round(4))
print('Testing Score:', knn_clf_fit.score(X_test, y_test).round(4))

# Generating Predictions
knn_clf_pred = knn_clf_fit.predict(X_test)

knn_clf_pred_probabilities = knn_clf_fit.predict_proba(X_test)



########################
# Machine Learning using logistic regression - Testing Score: 0.79
########################

got_data = got_copy.loc[: , ['male',
                             'book1_A_Game_Of_Thrones', 
                             'book3_A_Storm_Of_Swords',
                             'book4_A_Feast_For_Crows',
                             'numDeadRelations',
                             'o_dateOfBirth',
                             'popularity'
                             ]]

got_target =  got_copy.loc[: , 'isAlive']


X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.10,
            random_state = 508)

logreg = LogisticRegression(C = 0.1)


logreg_fit = logreg.fit(X_train, y_train)


# Predictions
logreg_pred = logreg_fit.predict(X_test)


print('Training Score', logreg_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_fit.score(X_test, y_test).round(4))



########################
# Random Forest - Testing Score: 0.84
########################

got_data = got_withdummies.loc[: , [  'male',
                                      'dateOfBirth',
                                      'numDeadRelations',
                                      'popularity',
                                      'culture_num',
                                      'house_num',
                                      'book1_A_Game_Of_Thrones',
                                      'book4_A_Feast_For_Crows',
                                      'book5_A_Dance_with_Dragons',
                                      'safe_cultures'
                                      ]]

got_target =  got_withdummies.loc[: , 'isAlive']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.1,
            random_state = 508)

rf_model = RandomForestClassifier(n_estimators = 500,
                                     criterion = 'entropy',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)


rf_fit = rf_model.fit(X_train, y_train)

# Getting scores
print('Training Score', rf_fit.score(X_train, y_train).round(4))
print('Testing Score:', rf_fit.score(X_test, y_test).round(4))

rf_pred_prob = rf_fit.predict_proba(X_test)[:, 1]
print("AUC score: {:.2f}".format(roc_auc_score(y_test, rf_pred_prob)))



########################
# Gradient Boosted Machines -  AUC: 0.86, Test Score: 0.86
########################

got_data = got_withdummies.loc[: , [ 
                                    'book1_A_Game_Of_Thrones',
                                    'book2_A_Clash_Of_Kings',
                                    'book4_A_Feast_For_Crows',
                                    'book5_A_Dance_with_Dragons',
                                    'dateOfBirth',
                                    'isNoble',
                                    'popularity',
                                    'o_house_num',
                                    'safe_cultures'
                                    ]]

got_target =  got_withdummies.loc[: , 'isAlive']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.1,
            random_state = 508)


gbm = GradientBoostingClassifier(loss = 'deviance',
                                  learning_rate = 1.34,
                                  n_estimators = 88,
                                  max_depth = 2,
                                  criterion = 'mse',
                                  warm_start = True,
                                  min_samples_leaf = 27,
                                  )

gbm_fit = gbm.fit(X_train, y_train)


gbm_predict = gbm_fit.predict(X_test)


# Training and Testing Scores
print('Training Score', gbm_fit.score(X_train, y_train).round(2))
print('Testing Score:', gbm_fit.score(X_test, y_test).round(2))

# AUC Score
gbm_pred_prob = gbm_fit.predict_proba(X_test)[:, 1]
print("AUC score: {:.2f}".format(roc_auc_score(y_test, gbm_pred_prob)))



########################
# Creating a confusion matrix
########################

print(confusion_matrix(y_true = y_test,
                       y_pred = gbm_predict))


# Visualizing a confusion matrix
labels = ['Dead', 'Alive']

cm = confusion_matrix(y_true = y_test,
                      y_pred = gbm_predict)


sns.heatmap(cm,
            annot = True,
            xticklabels = labels,
            yticklabels = labels,
            cmap = 'Blues',
            fmt = 'g')


plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix of the classifier')
plt.show()


########################
# Creating a classification report
########################

print(classification_report(y_true = y_test,
                            y_pred = gbm_predict))



# Changing the labels on the classification report
print(classification_report(y_true = y_test,
                            y_pred = gbm_predict,
                            target_names = labels))




###############################################################################
# Cross Validation with k-folds - Average AUC: 0.806
###############################################################################

# Cross Validating the gbm model with three folds
cv_gbm = cross_val_score(gbm_fit,
                         got_data,
                         got_target,
                         cv = 3,
                         scoring = 'roc_auc')


print(cv_gbm)


print(pd.np.mean(cv_gbm).round(3))

print('\nAverage: ',
      pd.np.mean(cv_gbm).round(3),
      '\nMinimum: ',
      min(cv_gbm).round(3),
      '\nMaximum: ',
      max(cv_gbm).round(3))



###############################################################################
# Storing Model Predictions and Summary
###############################################################################

# Storing predictions as a dictionary.
got_predictions_df = pd.DataFrame({
        'Actual' : y_test,
        'GBM Predictions': gbm_predict})

got_predictions_df.to_excel("GOT_Survival_Predictions.xlsx")


# End