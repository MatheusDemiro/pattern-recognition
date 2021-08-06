from algorithms.pre_processing.ClearData import PreProcessing

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV


class DecisionTree:
    def __init__(self):
        self._processing = PreProcessing()

    def algorithm(self):
        hyper = {'min_samples_split': [i for i in range(2, 50)],
                 'min_samples_leaf': [i * 0.1 for i in range(1, 6)],
                 'max_features': [i * 0.1 for i in range(1, 11)]}

        clf = DecisionTreeClassifier(random_state=0)

        labels = self._processing.getLabels()
        features = self._processing.getData("DT")

        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20,
                                                                                    random_state=0)

        # rs = RandomizedSearchCV(clf, hyper, random_state=0, cv=4, n_jobs=-1, iid=False)

        rs = GridSearchCV(clf, hyper, cv=4, n_jobs=-1, iid=False)

        rs.fit(features, labels)

        y_pred = rs.predict(features_test)

        print(confusion_matrix(labels_test, y_pred))

        recall = recall_score(labels_test, y_pred, average='micro')
        precision = precision_score(labels_test, y_pred, average='micro')
        fmeasure = f1_score(labels_test, y_pred, average='micro')

        print(rs.best_score_, rs.best_params_)

        return "Recall: %.4f\nPrecision: %.4f\nF-measure: %.4f" % (fmeasure, recall, precision)

    def execution(self):
        return self.algorithm()


DT = DecisionTree()

print(DT.execution())
