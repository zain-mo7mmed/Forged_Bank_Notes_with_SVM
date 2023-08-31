import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from dispaly_results_function import Display

path = "E:\\Artificial Intelligence\\SVM\\data_banknote_authentication.csv"
headers = ["variance", "skewness", "curtosis", "entropy", "class"]
bank_Data = pd.read_csv(path, names=headers, sep=",", header=None)

X = bank_Data.drop("class", axis=1)
y = bank_Data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# ---------------------------------------------------------------
# [Polynomial Kernel]
svc_poly = SVC(kernel='poly', degree=8)
svc_poly.fit(X_train, y_train)
y_pred_poly = svc_poly.predict(X_test)
print("Implementation of Polynomial")
cm_title_poly = "Confusion Matrix With Polynomial Kernel"
results_poly = Display(y_test, y_pred_poly, cm_title_poly)
results_poly.display_results()

# ---------------------------------------------------------------
# [Gaussian kernel]
svc_gaussian = SVC(kernel='rbf', degree=8)
svc_gaussian.fit(X_train, y_train)
y_pred_gaussian = svc_gaussian.predict(X_test)
print("Implementation of Gaussian")
cm_title_gaussian = "Confusion Matrix With Gaussian Kernel"
results_gaussian = Display(y_test, y_pred_gaussian, cm_title_gaussian)
results_gaussian.display_results()

# ---------------------------------------------------------------
# [Sigmoid kernel]
svc_sigmoid = SVC(kernel='sigmoid')
svc_sigmoid.fit(X_train, y_train)
y_pred_sigmoid = svc_sigmoid.predict(X_test)
print("Implementation of Sigmoid")
cm_title_sigmoid = "Confusion Matrix With Sigmoid Kernel"
results_sigmoid = Display(y_test, y_pred_sigmoid, cm_title_sigmoid)
results_sigmoid.display_results()
