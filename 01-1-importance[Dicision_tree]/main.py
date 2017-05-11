import pandas as pd
from sklearn import tree

PATH = 'data.csv'

class ImportanceTask(object):

    def __init__(self, path):
        """Загрузите выборку из файла titanic.csv с помощью пакета Pandas."""
        self.data = pd.read_csv(path, index_col='PassengerId')
        self.x = None
        self.y = None
        self.saved_columns = None

    def first(self):
        """Оставьте в выборке четыре признака: класс пассажира (Pclass), цену билета (Fare), возраст пассажира (Age) и его пол (Sex)."""
        self.saved_columns = ['Pclass', 'Fare', 'Age', 'Sex']
        self.x = self.data.ix[:, self.saved_columns]

    def second(self):
        """Обратите внимание, что признак Sex имеет строковые значения."""
        self.x['Sex'] = self.x['Sex'].apply(lambda x: 1 if x == 'male' else 0)

    def third(self):
        """Выделите целевую переменную — она записана в столбце Survived."""
        self.y = self.data['Survived']

    def forth(self):
        """Найдите все объекты, у которых есть пропущенные признаки, и удалите их из выборки."""
        self.x = self.x.dropna()
        self.y = self.y[self.x.index.values]

    def train_tree(self):
        dtc = tree.DecisionTreeClassifier(random_state=241)
        dtc.fit(self.x, self.y)
        importances = pd.Series(dtc.feature_importances_, index=self.saved_columns)
        with open("1.txt", "w") as f:
            f.write(" ".join(importances.sort_values(ascending=False).head(2).index.values))


    def __str__(self):
        return str(self.data)

if __name__ == '__main__':
    task = ImportanceTask(PATH)
    task.first()
    task.second()
    task.third()
    task.forth()
    task.train_tree()
    #print(task.y)
