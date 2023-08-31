import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

path = "E:\\Artificial Intelligence\\SVM\\data_banknote_authentication.csv"
headers = ["variance", "skewness", "curtosis", "entropy", "class"]
bank_Data = pd.read_csv(path, names=headers, sep=",", header=None)

X = bank_Data.drop("class", axis=1)
y = bank_Data["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

number_of_TRAIN_samples = X_train.shape[0]
number_of_TEST_samples = X_test.shape[0]
print(f"There are {number_of_TRAIN_samples} Samples For Training, And {number_of_TEST_samples} Samples For Testing")

# svc = SVC(kernel='linear', C=0.0001)
# svc.fit(X_train, y_train)
# y_pred = svc.predict(X_test)
#
# cm = confusion_matrix(y_test, y_pred)
# print('Report of Testing Data : ')
# print(classification_report(y_test, y_pred))
# # sns.heatmap(cm, annot=True, fmt='d').set_title('Confusion matrix of Hyperparameter SVM [Testing Data]')
# print('--------------------------------------------------------------------')
#
# y_pred_train = svc.predict(X_train)
# cm_train =confusion_matrix(y_train, y_pred_train)
# print('Report of Testing Data : ')
# print(classification_report(y_train, y_pred_train))
# sns.heatmap(cm_train, annot=True, fmt='d').set_title('Confusion matrix of Hyperparameter SVM [Training Data]')
# plt.show()

hyperparameter_dic = {'kernel': ['linear', 'rbf'],
                      'C': [0.0001, 1, 10],
                      'gamma': [1, 10, 100]}
svc = SVC()
grid_search = GridSearchCV(svc, hyperparameter_dic, scoring='f1', return_train_score=True, cv=5, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_parameters = grid_search.best_params_
best_f1 = grid_search.best_score_
print('\n--------------------------------------------------------------------\n')
print('The best model was:', best_model)
print('The best parameter values were:', best_parameters)
print('The best f1-score was:', best_f1)

gs_test_scores = grid_search.cv_results_['mean_test_score']
gs_train_scores = grid_search.cv_results_['mean_train_score']
print('\n--------------------------------------------------------------------\n')
print('Test f1-Scores is : ', gs_test_scores)
print('Train f1-Scores is : ', gs_train_scores)
print('\n--------------------------------------------------------------------\n')
print('So, The Best f1-score of Testing Data is : ', max(gs_test_scores))
print('And The Best f1-score of Testing Data is : ', max(gs_train_scores))
print('\n--------------------------------------------------------------------\n')

svc = SVC(kernel='rbf', C=1, gamma=1)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('Report of Testing Data : ')
print(classification_report(y_test, y_pred))
sns.heatmap(cm, annot=True, fmt='d').set_title('Confusion matrix of Hyperparameter SVM [Testing Data]')
plt.show()
