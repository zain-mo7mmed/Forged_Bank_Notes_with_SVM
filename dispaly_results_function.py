from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class Display:
    def __init__(self, y_test, y_pred, cm_title):
        self.y_test = y_test
        self.y_pred = y_pred
        self.cm_title = cm_title

    def display_results(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d').set_title(self.cm_title)
        print(classification_report(self.y_test, self.y_pred))
        plt.show()
        print('\n-----------------------------------------------------\n')
