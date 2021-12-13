#!/usr/bin/env python
# coding: utf-8

# In[6]:


import sklearn
import sklearn.datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Source -> Precept 6: Classification Methods
from sklearn.model_selection import train_test_split
# Source -> Precept 7: Decision Tree
from sklearn.tree import DecisionTreeClassifier
# Source -> https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
from sklearn.metrics import confusion_matrix
# Source -> https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
from sklearn.metrics import f1_score
# Import the linear regression model
from sklearn.linear_model import LinearRegression
# Source -> Precept 6: Classification Methods
from sklearn.ensemble import RandomForestClassifier
# Single Variable Linear Model
from sklearn.metrics import mean_squared_error
# Source -> Precept 6: Classification Methods
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# Source -> https://inblog.in/Feature-Importance-in-Naive-Bayes-Classifiers-5qob5d5sFW
from sklearn.inspection import permutation_importance
# Source -> Precept 7: Decision Tree
from sklearn import tree


# In[3]:


# read input csv
df = pd.read_csv('cps.csv')

# remove unwanted columns
data_flags = df[['QAGE', 'QAGE_SP', 'QMARST', 'QMARST_SP', 'QSEX', 'QSEX_SP', 'QRACE', 'QRACE_SP', 'QVETSTAT', 'QVETSTAT_SP', 'QNATIVIT', 'QNATIVIT_SP', 'QYRIMMIG', 'QYRIMMIG_SP', 'QHISPAN', 'QHISPAN_SP', 'QCLASSWK', 'QCLASSWK_SP', 'QUHRSWORK1', 'QUHRSWORK1_SP', 'QEMPSTAT', 'QEMPSTAT_SP', 'QIND', 'QIND_SP', 'QLABFORC', 'QLABFORC_SP', 'QOCC', 'QOCC_SP', 'QEDUC', 'QEDUC_SP', 'QSCHCOL1', 'QSCHCOL1_SP', 'QSCHCOL2', 'QSCHCOL2_SP', 'QSCHCOL3', 'QSCHCOL3_SP']]
df = df.drop(columns=['ASECFLAG', 'HFLAG', 'PERNUM', 'ASECWTH', 'PERNUM_SP', 'QAGE', 'QAGE_SP', 'QMARST', 'QMARST_SP', 'QSEX', 'QSEX_SP', 'QRACE', 'QRACE_SP', 'QVETSTAT', 'QVETSTAT_SP', 'QNATIVIT', 'QNATIVIT_SP', 'QYRIMMIG', 'QYRIMMIG_SP', 'QHISPAN', 'QHISPAN_SP', 'QCLASSWK', 'QCLASSWK_SP', 'QUHRSWORK1', 'QUHRSWORK1_SP', 'QEMPSTAT', 'QEMPSTAT_SP', 'QIND', 'QIND_SP', 'QLABFORC', 'QLABFORC_SP', 'QOCC', 'QOCC_SP', 'QEDUC', 'QEDUC_SP', 'QSCHCOL1', 'QSCHCOL1_SP', 'QSCHCOL2', 'QSCHCOL2_SP', 'QSCHCOL3', 'QSCHCOL3_SP'])
# replace NA's with -1
df = df.fillna(value = -1)
# Source -> https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.filter.html
spouse = df.filter(like='_SP')
interviewee = df.drop(columns=spouse.columns)


# In[3]:


df.head()


# In[4]:


df.corr()


# In[94]:


# Source -> https://stackoverflow.com/questions/17812978/how-to-plot-two-columns-of-a-pandas-data-frame-using-points
df.plot(x='SEX', y='ASECWT', style='o')
# Wealth based on sex
print((df[df['SEX'] == 1.0]['ASECWT']).mean())
print((df[df['SEX'] == 2.0]['ASECWT']).mean())


# In[78]:


# https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
X_gd = df['AGE']
Y_gd = df['ASECWT']
# https://stackoverflow.com/questions/53723928/attributeerror-series-object-has-no-attribute-reshape
X_gd = X_gd.values.reshape(-1, 1)
Y_gd = Y_gd.values.reshape(-1, 1)

lm = LinearRegression()
lm.fit(X=X_gd, y=Y_gd)

# Wealth based on age
print(lm.coef_)
print(lm.intercept_)
b1 = lm.coef_[0]
b0 = lm.intercept_
plt.plot(X_gd, Y_gd, 'o')
plt.plot(X_gd, b1*X_gd + b0)


# In[97]:


# Wealth based on race
df.plot(x='RACE', y='ASECWT', style='o')
print((df[df['RACE'] == 100]['ASECWT']).mean()) # White
print((df[df['RACE'] == 200]['ASECWT']).mean()) # Black
print((df[df['RACE'] == 300]['ASECWT']).mean()) # American Indian
print((df[df['RACE'] == 651]['ASECWT']).mean()) # Asian
print((df[df['RACE'] > 800]['ASECWT']).mean()) # Mixed


# In[69]:


# Wealth based on citizenship status
df.plot(x='CITIZEN', y='ASECWT', style='o')


# In[70]:


# Wealth based on ethnicity
df.plot(x='HISPAN', y='ASECWT', style='o')


# In[81]:


# Wealth based on occupation
df.plot(x='OCC', y='ASECWT', style='o')


# In[82]:


# Wealth based on industry
df.plot(x='IND', y='ASECWT', style='o')


# In[83]:


# Wealth based on education
df.plot(x='EDUC', y='ASECWT', style='o')


# In[35]:


# Label based on income
# Source -> https://stackoverflow.com/questions/58545136/adding-column-to-pandas-data-frame-that-provides-labels-based-on-condition
interviewee['label'] = [1 if x > 1800 else 0 for x in interviewee['ASECWT']] # 1 for target salary
spouse['label'] = [1 if x > 1800 else 0 for x in spouse['ASECWT_SP']] # 1 for target salary


# In[54]:


X = interviewee.drop(columns=['ASECWT','label'])
Y = interviewee['label']

# train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

X2 = spouse.drop(columns=['ASECWT_SP', 'label'])
Y2 = spouse['label']

# train-test split
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, Y2, test_size=0.2)


# In[ ]:


# Predict wealth from remaining features
clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)
pred_train = clf.predict(X_train)
pred_test = clf.predict(X_test)
print('Decision Tree Train Accuracy: {}'.format(np.sum(pred_train==Y_train)/len(Y_train)))
print('Decision Tree Train F-score: {}'.format(f1_score(Y_train, pred_train)))
print('Decision Tree Test Accuracy: {}'.format(np.sum(pred_test==Y_test)/len(Y_test)))
print('Decision Tree Test F-score: {}'.format(f1_score(Y_test, pred_test)))
print(confusion_matrix(Y_train, pred_train))
print(confusion_matrix(Y_test, pred_test))


# In[52]:


# Repeat for spouse
clf = DecisionTreeClassifier()
clf.fit(X_train2, Y_train2)
pred_train2 = clf.predict(X_train2)
pred_test2 = clf.predict(X_test2)
print('Decision Tree Train Accuracy: {}'.format(np.sum(pred_train2==Y_train2)/len(Y_train2)))
print('Decision Tree Train F-score: {}'.format(f1_score(Y_train2, pred_train2)))
print('Decision Tree Test Accuracy: {}'.format(np.sum(pred_test2==Y_test2)/len(Y_test2)))
print('Decision Tree Test F-score: {}'.format(f1_score(Y_test2, pred_test2)))
print(confusion_matrix(Y_train2, pred_train2))
print(confusion_matrix(Y_test2, pred_test2))


# In[56]:


# Important features for Decision Tree
print(clf.feature_importances_)
feature1 = X.iloc[:, 0].name
feature2 = X.iloc[:, 2].name
feature3 = X.iloc[:, 25].name
print(feature1)
print(feature2)
print(feature3)


# In[12]:


rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
pred_train = rf.predict(X_train)
pred_test = rf.predict(X_test)
print('Random Forest Train Accuracy: {}'.format(np.sum(pred_train==Y_train)/len(Y_train)) )
print('Random Forest Train F-score: {}'.format(f1_score(Y_train, pred_train)))
print('Random Forest Test Accuracy: {}'.format(np.sum(pred_test==Y_test)/len(Y_test)) )
print('Random Forest Test F-score: {}'.format(f1_score(Y_test, pred_test)))
print(confusion_matrix(Y_train, pred_train))
print(confusion_matrix(Y_test, pred_test))


# In[13]:


rf = RandomForestClassifier()
rf.fit(X_train2, Y_train2)
pred_train2 = rf.predict(X_train2)
pred_test2 = rf.predict(X_test2)
print('Random Forest Train Accuracy: {}'.format(np.sum(pred_train2==Y_train2)/len(Y_train2)) )
print('Random Forest Train F-score: {}'.format(f1_score(Y_train2, pred_train2)))
print('Random Forest Test Accuracy: {}'.format(np.sum(pred_test2==Y_test2)/len(Y_test2)) )
print('Random Forest Test F-score: {}'.format(f1_score(Y_test2, pred_test2)))
print(confusion_matrix(Y_train2, pred_train2))
print(confusion_matrix(Y_test2, pred_test2))


# In[ ]:


# Important features for Random Forest Classifier
# Source -> https://mljar.com/blog/feature-importance-in-random-forest/
print(rf.feature_importances_)
feature1 = cancer_df.iloc[:, 20].name
feature2 = cancer_df.iloc[:, 22].name
feature3 = cancer_df.iloc[:, 27].name
print(feature1)
print(feature2)
print(feature3)


# In[57]:


knn = KNeighborsClassifier(weights='distance')
nb = MultinomialNB()
lgr = LogisticRegression()
svm = SVC()

models = [knn, nb, lgr, svm]
names = ['Nearest Neighbors', 'Naive Bayes', 'Logistic Regression', 'Support Vector Machine']

## ignore convergence warning from logistic regression & svm
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

for model, name in zip(models, names):
    model.fit(X_train, Y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    print(name + ' Train Accuracy: {}'.format(np.sum(pred_train==Y_train)/len(Y_train)) )
    print(name + ' Train F-score: {}'.format(f1_score(Y_train, pred_train)))
    print(name + ' Test Accuracy: {}'.format(np.sum(pred_test==Y_test)/len(Y_test)) )
    print(name + ' Test F-score: {}'.format(f1_score(Y_test, pred_test)))
    print(confusion_matrix(Y_train, pred_train))
    print(confusion_matrix(Y_test, pred_test))


# In[ ]:


# Feature importance for K-Nearest Neighbors
features = X.columns
imps = permutation_importance(knn, X_test, Y_test)
importances = imps.importances_mean
std = imps.importances_std
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X_test.shape[1]):
    print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))


# In[60]:


# Feature importance for Naive Bayes
features = X.columns
imps = permutation_importance(nb, X_test, Y_test)
importances = imps.importances_mean
std = imps.importances_std
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X_test.shape[1]):
    print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))


# In[62]:


# Important features for Logistic Regression
print(lgr.coef_)
feature1 = X.iloc[:, 3].name
feature2 = X.iloc[:, 4].name
feature3 = X.iloc[:, 1].name
print(feature1)
print(feature2)
print(feature3)


# In[ ]:


# Feature importance for Support Vector Machines
features = X.columns
imps = permutation_importance(svm, X_test, Y_test)
importances = imps.importances_mean
std = imps.importances_std
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X_test.shape[1]):
    print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))


# In[25]:


# Label based on position
interviewee['label'] = [1 if x < 38 else 0 for x in interviewee['OCC']] # 1 for target salary
spouse['label'] = [1 if x < 38 else 0 for x in spouse['OCC_SP']] # 1 for target salary

# Manipulate columns based on desired prediction
X = interviewee.drop(columns=['OCC2010','LABFORCE','UHRSWORKT','CLASSWKR', 'IND', 'OCC', 'label'])
Y = interviewee['label']

# train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

X2 = spouse.drop(columns=['OCC2010_SP','LABFORCE_SP','UHRSWORKT_SP','CLASSWKR_SP', 'IND_SP', 'OCC_SP', 'label'])
Y2 = spouse['label']

# train-test split
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, Y2, test_size=0.2)


# In[23]:


# Predict occupation based on remaining features
clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)
pred_train = clf.predict(X_train)
pred_test = clf.predict(X_test)
print('Decision Tree Train Accuracy: {}'.format(np.sum(pred_train==Y_train)/len(Y_train)))
print('Decision Tree Train F-score: {}'.format(f1_score(Y_train, pred_train)))
print('Decision Tree Test Accuracy: {}'.format(np.sum(pred_test==Y_test)/len(Y_test)))
print('Decision Tree Test F-score: {}'.format(f1_score(Y_test, pred_test)))
print(confusion_matrix(Y_train, pred_train))
print(confusion_matrix(Y_test, pred_test))


# In[26]:


# Repeat for spouse
clf = DecisionTreeClassifier()
clf.fit(X_train2, Y_train2)
pred_train2 = clf.predict(X_train2)
pred_test2 = clf.predict(X_test2)
print('Decision Tree Train Accuracy: {}'.format(np.sum(pred_train2==Y_train2)/len(Y_train2)))
print('Decision Tree Train F-score: {}'.format(f1_score(Y_train2, pred_train2)))
print('Decision Tree Test Accuracy: {}'.format(np.sum(pred_test2==Y_test2)/len(Y_test2)))
print('Decision Tree Test F-score: {}'.format(f1_score(Y_test2, pred_test2)))
print(confusion_matrix(Y_train2, pred_train2))
print(confusion_matrix(Y_test2, pred_test2))


# In[ ]:


# Decision Tree Feature Importance
print(clf.feature_importances_)
feature1 = X.iloc[:, 17].name
feature2 = X.iloc[:, 18].name
feature3 = X.iloc[:, 19].name
print(feature1)
print(feature2)
print(feature3)


# In[ ]:


# Source -> https://stackoverflow.com/questions/59447378/sklearn-plot-tree-plot-is-too-small
plt.figure(figsize=(16,10))  # set plot size (denoted in inches)
tree.plot_tree(clf, fontsize=10)
# Visualize Decision Tree
plt.show()


# In[24]:


# Predict job based on other features
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
pred_train = rf.predict(X_train)
pred_test = rf.predict(X_test)
print('Random Forest Train Accuracy: {}'.format(np.sum(pred_train==Y_train)/len(Y_train)) )
print('Random Forest Train F-score: {}'.format(f1_score(Y_train, pred_train)))
print('Random Forest Test Accuracy: {}'.format(np.sum(pred_test==Y_test)/len(Y_test)) )
print('Random Forest Test F-score: {}'.format(f1_score(Y_test, pred_test)))
print(confusion_matrix(Y_train, pred_train))
print(confusion_matrix(Y_test, pred_test))


# In[ ]:


# Repeat for spouse
rf = RandomForestClassifier()
rf.fit(X_train2, Y_train2)
pred_train2 = rf.predict(X_train2)
pred_test2 = rf.predict(X_test2)
print('Random Forest Train Accuracy: {}'.format(np.sum(pred_train2==Y_train2)/len(Y_train2)) )
print('Random Forest Train F-score: {}'.format(f1_score(Y_train2, pred_train2)))
print('Random Forest Test Accuracy: {}'.format(np.sum(pred_test2==Y_test2)/len(Y_test2)) )
print('Random Forest Test F-score: {}'.format(f1_score(Y_test2, pred_test2)))
print(confusion_matrix(Y_train2, pred_train2))
print(confusion_matrix(Y_test2, pred_test2))


# In[33]:


# Random Forest Feature Importance
print(rf.feature_importances_)
feature1 = X.iloc[:, 18].name
feature2 = X.iloc[:, 19].name
feature3 = X.iloc[:, 20].name
print(feature1)
print(feature2)
print(feature3)


# In[111]:


# Linear Regression to predict wealth
X = interviewee['OCC'].values
X = X[:,np.newaxis]
Y = interviewee['ASECWT'].values
Y = Y[:,np.newaxis]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

lm = LinearRegression()
lm.fit(X = X, y = Y)

print(lm.coef_)
print(lm.intercept_)
b1 = lm.coef_[0]
b0 = lm.intercept_
plt.plot(X, Y, 'o')
plt.plot(X, b1*X + b0)

Y_pred = lm.predict(X_test)
plt.scatter(Y_test, Y_pred)
plt.plot(Y_test, Y_test, color="black")

print('Mean Squared Error: ')
print(mean_squared_error(Y_test, Y_pred))
print('R^2 score on test data: {:.5f}'.format(lm.score(X_test, Y_test)) )


# In[112]:


# Multivariable Linear Model
X = np.column_stack((interviewee['OCC'], interviewee['IND'], interviewee['SEX'], interviewee['RACE']))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

lm = LinearRegression()
lm.fit(X = X, y = Y)

print(lm.coef_)
print(lm.intercept_)
b1 = lm.coef_[0]
b0 = lm.intercept_
plt.plot(X, Y, 'o')

Y_pred = lm.predict(X_test)
plt.scatter(Y_test, Y_pred)
plt.plot(Y_test, Y_test, color="black")

print('Mean Squared Error: ')
print(mean_squared_error(Y_test, Y_pred))
print('R^2 score on test data: {:.5f}'.format(lm.score(X_test, Y_test)) )


# In[ ]:




