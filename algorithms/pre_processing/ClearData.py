import pandas as pd
from sklearn import preprocessing as prep
from sklearn.feature_selection import VarianceThreshold


class PreProcessing:
    def __init__(self):
        self._path = "C:/Users/User/PycharmProjects/rp_2019_1/dataset/data.csv"
        self._data = pd.read_csv(self._path)
        self._labels = self._data.classe

    def getPath(self):
        return self._path

    def setPath(self, path):
        self._path = path

    def getData(self, estimator):
        return self.clearData(estimator)

    def getTargets(self, key):
        # target = {0:'andando', 1:'bebendo', 2:'comendo', 3:'ocio', 4:'ruminando'}
        target = {0: 'comendo', '1': 'ócio', '2': 'ruminando'}
        return target[key]

    def getLabels(self):
        return self._labels

    def normalize(self, data):
        p = prep.MaxAbsScaler()
        p.fit_transform(data)
        return p.transform(data)

    def featureSelection(self):
        # Attribute Selection
        attribute_selection = VarianceThreshold(threshold=(.8 * (1 - .8)))
        new_data = attribute_selection.fit_transform(self._data.drop(columns="classe"))

        return new_data

    def clearData(self, estimator):
        self.convertType()
        attr = self._data.get_values()
        for i in range(288):
            if estimator == "NB":
                self._data.loc[i, 'lat'] = float(attr[i][0] * -1)
                self._data.loc[i, 'long'] = float(attr[i][1] * -1)
            self._data.loc[i, 'tbs-pasto'] = int(attr[i][2]) / 1000
            self._data.loc[i, 'ur-pasto'] = int(attr[i][3]) / 1000
            # self._data.loc[i, 'tgn-pasto'] = int(attr[i][4]) / 1000
            self._data.loc[i, 'tpo-pasto'] = int(attr[i][5]) / 1000
            self._data.loc[i, 'tbs-sombra'] = int(attr[i][6]) / 1000
            self._data.loc[i, 'ur-sombra'] = int(attr[i][7]) / 1000
            # self._data.loc[i, 'tgn-sombra'] = int(attr[i][8]) / 1000
            self._data.loc[i, 'tpo-sombra'] = int(attr[i][9]) / 1000
        return self.normalize(self.featureSelection())

    def enconder(self):
        le = prep.LabelEncoder()
        labels = self.getLabels()
        # Return non-numeric labels values: le.inverse_transform(labels)
        le.fit(labels)
        return le.transform(labels)

    def convertType(self):
        '''self._data.astype(
            {'lat': 'float64', 'long': 'float64', 'tbs-pasto': 'float64', 'ur-pasto': 'float64', 'tgn-pasto': 'float64',
             'tpo-pasto': 'float64', 'tbs-sombra': 'float64', 'ur-sombra': 'float64', 'tgn-sombra': 'float64',
             'tpo-sombra': 'float64', 'posicao': 'int64', 'local': 'int64', 'classe': 'str'})'''
        self._data.astype(
            {'lat': 'float64', 'long': 'float64', 'tbs-pasto': 'float64', 'ur-pasto': 'float64',
             'tpo-pasto': 'float64', 'tbs-sombra': 'float64', 'ur-sombra': 'float64',
             'tpo-sombra': 'float64', 'posicao': 'int64', 'local': 'int64', 'classe': 'str'})
