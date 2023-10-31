# importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# loading the dataset
data = pd.read_csv('combined_dataset.csv')


# heatmap
sns.heatmap(data[['ranking', 'activeDuration', 'urlLen', 'domainLen', 'nosOfSubdomain']].corr(), annot = True)


# dropping the feature, domainLen
data.drop('domainLen', axis = 1, inplace = True)


# box-plot
plt.figure(figsize = (15, 15))
plt.subplot(2, 2, 1)
sns.boxplot(x = 'label', y = 'ranking', data = data)
plt.subplot(2, 2, 2)
sns.boxplot(x = 'label', y = 'activeDuration', data = data)
plt.subplot(2, 2, 3)
sns.boxplot(x = 'label', y = 'urlLen', data = data)
plt.subplot(2, 2, 4)
sns.boxplot(x = 'label', y = 'nosOfSubdomain', data = data)
plt.show()


# getting all the features in X and target label in y
X = data.iloc[:, 1:-1]
y = data['label']


# train-test-split of the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1234)


# feature scaling
scaler = StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# kNN Model Training using Grid Search for hyper-parameter tuning
knn_model = KNeighborsClassifier()
params = {'n_neighbors' : [2, 3, 4, 5, 6, 7, 8, 9, 10]}
grid_search = GridSearchCV(knn_model, param_grid = params, cv = 5, verbose = 3).fit(X_train_scaled, y_train)

print('Best Hyper-parameter:', grid_search.best_params_)
n_neighbors = grid_search.best_params_['n_neighbors']

knn_model = KNeighborsClassifier(n_neighbors = n_neighbors).fit(X_train_scaled, y_train)


# getting the test predictions, test probabilities and test auc score
y_test_pred = knn_model.predict(X_test_scaled)
y_test_prob = knn_model.predict_proba(X_test_scaled)

test_accuracy = metrics.accuracy_score(y_test, y_test_pred)

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_prob[:,1])
roc_auc = metrics.auc(fpr, tpr)

test_auc = roc_auc


# getting the Test Accuracy and Test AUC Score
print("Test Accuracy: ", test_accuracy)
print("Test AUC: ", test_auc)

# Getting the Test Precicion, Recall, F1-Score and Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_test_pred)
TP = cm[1,1] # true positive 
TN = cm[0,0] # true negatives
FP = cm[0,1] # false positives
FN = cm[1,0] # false negatives

precision = TP/(TP+FP)
recall = TP/(TP+FN)
f1 = (2*precision*recall)/(precision+recall)

print('Testing Precision: ', precision)
print('Testing Recall: ', recall)
print('Testing F1-Score: ', f1)
print('Testing Confusion Matrix: ')
print(cm)
print('\n')


# displaying the ROC Curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_prob[:,1],
                                         drop_intermediate = False)
auc_score = metrics.roc_auc_score(y_test, y_test_prob[:,1])
plt.figure(figsize=(5, 5))
plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()



# Logistic Regression training using hyper-parameter tuning
lr_model = LogisticRegression()
params = {'C': [0.01, 0.1, 1, 10, 100]}

grid_search = GridSearchCV(lr_model, param_grid = params, cv = 5, verbose = 3).fit(X_train_scaled, y_train)

print('Best Hyperparameter:', grid_search.best_params_)
best_C = grid_search.best_params_['C']

lr_model = LogisticRegression(C = best_C).fit(X_train_scaled, y_train)


# getting the test predictions, test probabilities and test auc score
y_test_pred = lr_model.predict(X_test_scaled)
y_test_prob = lr_model.predict_proba(X_test_scaled)

test_accuracy = metrics.accuracy_score(y_test, y_test_pred)

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_prob[:,1])
roc_auc = metrics.auc(fpr, tpr)

test_auc = roc_auc


# getting the Test Accuracy and Test AUC Score
print("Test Accuracy: ", test_accuracy)
print("Test AUC: ", test_auc)

# Getting the Test Precicion, Recall, F1-Score and Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_test_pred)
TP = cm[1,1] # true positive 
TN = cm[0,0] # true negatives
FP = cm[0,1] # false positives
FN = cm[1,0] # false negatives

precision = TP/(TP+FP)
recall = TP/(TP+FN)
f1 = (2*precision*recall)/(precision+recall)

print('Testing Precision: ', precision)
print('Testing Recall: ', recall)
print('Testing F1-Score: ', f1)
print('Testing Confusion Matrix: ')
print(cm)
print('\n')


# displaying the ROC Curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_prob[:,1],
                                         drop_intermediate = False)
auc_score = metrics.roc_auc_score(y_test, y_test_prob[:,1])
plt.figure(figsize=(5, 5))
plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()



# Gradient Boosting Classifier training using Grid Search for hyper-parameter tuning
svc_model = SVC(random_state = 1234)
params = {'C' : [0.1, 1, 10], 'kernel' : ['linear', 'rbf']}

grid_search = GridSearchCV(svc_model, param_grid = params, cv = 5, verbose = 3).fit(X_train_scaled, y_train)

print('Best Hyperparameter(s):', grid_search.best_params_)
C = grid_search.best_params_['C']
kernel = grid_search.best_params_['kernel']

svc_model = SVC(C = C, 
                kernel = kernel, 
                random_state = 1234,
                probability = True).fit(X_train_scaled, y_train)


# getting the test predictions, test probabilities and test auc score
y_test_pred = svc_model.predict(X_test_scaled)
y_test_prob = svc_model.predict_proba(X_test_scaled)

test_accuracy = metrics.accuracy_score(y_test, y_test_pred)

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_prob[:,1])
roc_auc = metrics.auc(fpr, tpr)

test_auc = roc_auc


# getting the Test Accuracy and Test AUC Score
print("Test Accuracy: ", test_accuracy)
print("Test AUC: ", test_auc)

# Getting the Test Precicion, Recall, F1-Score and Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_test_pred)
TP = cm[1,1] # true positive 
TN = cm[0,0] # true negatives
FP = cm[0,1] # false positives
FN = cm[1,0] # false negatives

precision = TP/(TP+FP)
recall = TP/(TP+FN)
f1 = (2*precision*recall)/(precision+recall)

print('Testing Precision: ', precision)
print('Testing Recall: ', recall)
print('Testing F1-Score: ', f1)
print('Testing Confusion Matrix: ')
print(cm)
print('\n')


# displaying the ROC Curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_prob[:,1],
                                         drop_intermediate = False)
auc_score = metrics.roc_auc_score(y_test, y_test_prob[:,1])
plt.figure(figsize=(5, 5))
plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()