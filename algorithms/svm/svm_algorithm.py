from algorithms.pre_processing.ClearData import PreProcessing

from sklearn.svm import SVC
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV


class SVM:
    def __init__(self):
        self._processing = PreProcessing()

    def algorithm(self):
        hyper = {
            "C": [1.0, 2.0, 3.0],
            "kernel": ['linear', 'rbf', 'sigmoid', 'poly'],
            "degree": [2, 3, 4],
            "gamma": ["scale"]
        }

        labels = self._processing.getLabels()
        features = self._processing.clearData("SVM")

        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30,
                                                                                    random_state=0)

        clf = SVC(random_state=0)

        rs = RandomizedSearchCV(clf, hyper, cv=4, n_jobs=-1, iid=False, verbose=1000)

        rs.fit(features_train, labels_train)

        y_pred = rs.predict(features_test)

        print(confusion_matrix(labels_test, y_pred))

        recall = recall_score(labels_test, y_pred, average='micro')
        precision = precision_score(labels_test, y_pred, average='micro')
        fmeasure = f1_score(labels_test, y_pred, average='micro')

        print(rs.best_params_)

        return "Recall: %.4f\nPrecision: %.4f\nF-measure: %.4f" % (fmeasure, recall, precision)

    def execution(self):
        return self.algorithm()


SVM = SVM()

print(SVM.execution())
