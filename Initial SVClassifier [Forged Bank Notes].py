import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

path = "E:\\Artificial Intelligence\\SVM\\data_banknote_authentication.csv"
headers = ["variance", "skewness", "curtosis", "entropy", "class"]
bank_Data = pd.read_csv(path, names=headers, sep=",", header=None)

# sns.pairplot(bank_Data, hue='class')
# plt.show()

X = bank_Data.drop("class", axis=1)
y = bank_Data["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

number_of_TRAIN_samples = X_train.shape[0]
number_of_TEST_samples = X_test.shape[0]
print(f"There are {number_of_TRAIN_samples} Samples For Training, And {number_of_TEST_samples} Samples For Testing")

svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('Report of Testing Data : ')
print(classification_report(y_test, y_pred))
# sns.heatmap(cm, annot=True, fmt='d').set_title('Confusion matrix of linear SVM [Testing Data]')
print('--------------------------------------------------------------------')

y_pred_train = svc.predict(X_train)

cm_train = confusion_matrix(y_train, y_pred_train)
print('Report of Training Data : ')
print(classification_report(y_train, y_pred_train))
sns.heatmap(cm_train, annot=True, fmt='d').set_title('Confusion matrix of linear SVM [Training Data]')

plt.show()
