from algorithms.pre_processing.ClearData import PreProcessing

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV


class RandomForest:
    def __init__(self):
        self._processing = PreProcessing()

    def algorithm(self):
        min_samples_split = [i for i in range(2, 51)]
        min_samples_leaf = [i * 0.1 for i in range(1, 5)]
        max_features = [i * 0.1 for i in range(1, 11)]

        hyper = {
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features
        }

        labels = self._processing.getLabels()
        features = self._processing.getData("RF")

        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33,
                                                                                    random_state=0)
        random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=0)

        gs = GridSearchCV(random_forest_clf, cv=4, param_grid=hyper, iid=False)

        gs.fit(features_train, labels_train)

        y_pred = gs.predict(features_test)

        recall = recall_score(labels_test, y_pred, average='micro')
        precision = precision_score(labels_test, y_pred, average='micro')
        fmeasure = f1_score(labels_test, y_pred, average='micro')

        print(confusion_matrix(labels_test, y_pred, labels=[0, 1, 2, 3, 4]))

        print(gs.best_params_)

        return "Recall: %.4f\nPrecision: %.4f\nF-measure: %.4f" % (fmeasure, recall, precision)

    def execution(self):
        return self.algorithm()


RFC = RandomForest()

print(RFC.execution())
