from algorithms.pre_processing.ClearData import PreProcessing

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, recall_score, precision_score


class NaiveBayes:
    def __init__(self):
        self._processing = PreProcessing()

    def algorithm(self):
        clf = MultinomialNB()

        labels = self._processing.getLabels()
        features = self._processing.getData("NB")

        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20, random_state=0)

        clf.fit(features_train, labels_train)

        y_pred = clf.predict(features_test)

        recall = recall_score(labels_test, y_pred, average='micro')
        precision = precision_score(labels_test, y_pred, average='micro')
        fmeasure = f1_score(labels_test, y_pred, average='micro')

        return "Recall: %.4f\nPrecision: %.4f\nF-measure: %.4f"%(recall, precision, fmeasure)

    def execution(self):
        return self.algorithm()

NB = NaiveBayes()
    
print(NB.execution())
