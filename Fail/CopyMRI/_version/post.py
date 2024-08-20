import seaborn as sns
import matplotlib.pyplot as plt
from Utils import *
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split

arrs = []
for name in ["x.npy", "y.npy", "x_test.npy", "y_test.npy"]:
    with open(name, 'rb') as f:
        a = np.load(f)
        ut.over(a)
        arrs.append(a)


x, y, x_test, y_test = arrs
# x, x_val, y, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
clf = LinearSVC(random_state=42, C=1e-4)
ut.over(x, "x")
ut.over(y, "y")
clf.fit(x, y)
print(clf.score(x_test, y_test))
# print(clf.score(x_val, y_val))
print(clf.score(x, y))


def show_heatmap(clf, x_test, y_test):
    y_pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    # Create a heatmap using seaborn
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
    # Add labels and title
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


show_heatmap(clf, x_test, y_test)
