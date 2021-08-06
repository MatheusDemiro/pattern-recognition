from algorithms.pre_processing.ClearData import PreProcessing

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV


class KNN:
    def __init__(self):
        self._processing = PreProcessing()
        self._NUM_TRIALS = 30

    def algorithm(self):
        hyper = {"n_neighbors": range(1,31)}

        labels = self._processing.getLabels()
        features = self._processing.getData("KNN")

        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.33, random_state=0)

        sss.get_n_splits(features, labels)

        # features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20,
        #                                                                             random_state=0)

        clf = KNeighborsClassifier()

        precision_scores = []
        recall_scores = []
        f1_scores = []

        sss = StratifiedShuffleSplit(n_splits=10)
        for train_index, test_index in sss.split(features, labels):
            x_train, x_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            rs = GridSearchCV(clf, hyper, cv=4, n_jobs=-1, iid=False)

            rs.fit(x_train, y_train)
            y_pred = rs.predict(x_test)

            precision_scores.append(precision_score(y_test, y_pred, average='micro'))
            recall_scores.append(recall_score(y_test, y_pred, average='micro'))
            f1_scores.append(f1_score(y_test, y_pred, average='micro'))

            print(confusion_matrix(y_test, y_pred))

        #rs = RandomizedSearchCV(clf, hyper, random_state=0, cv=4, n_jobs=-1, iid=False)

        # rs = GridSearchCV(clf, hyper, cv=4, n_jobs=-1, iid=False)
        #
        # rs.fit(features_train, labels_train)
        #
        # y_pred = rs.predict(features_test)
        #
        # recall = recall_score(labels_test, y_pred, average='micro')
        # precision = precision_score(labels_test, y_pred, average='micro')
        # fmeasure = f1_score(labels_test, y_pred, average='micro')

        # print(confusion_matrix(labels_test, y_pred, labels=[0,1,2,3,4]))

        # print(rs.best_params_)

        return "Recall: %.4f\nPrecision: %.4f\nF-measure: %.4f"%(sum(f1_scores)/len(f1_scores),
                                                                 sum(recall_scores)/len(recall_scores), sum(precision_scores)/len(precision_scores))

    def execution(self):
        return self.algorithm()

KNN = KNN()

print(KNN.execution())