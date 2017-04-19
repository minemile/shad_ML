import pandas as pd

# Решение задач 1 недели. Работа с пандас и данными Титаника

def create_answer(filename, answer):
    with open(filename, 'w') as f:
        f.write(answer)

def create_dataFrame(filepath='data.csv'):
    return pd.read_csv('data.csv', index_col='PassengerId')

def first(data):
    """Какое количество мужчин и женщин ехало на корабле?"""
    count_sex = data['Sex'].value_counts()
    answer = str(count_sex['male']) + " " + str(count_sex['female'])
    create_answer('1.txt', answer)

def second(data):
    """Какой части пассажиров удалось выжить? Посчитайте долю выживших пассажиров."""
    count_survived = data['Survived'].value_counts()
    answer = "{:0.2f}".format(100 * count_survived[1] / count_survived.sum())
    create_answer('2.txt', answer)

def third(data):
    """Какую долю пассажиры первого класса составляли среди всех пассажиров?"""
    count_first_class = data['Pclass'].value_counts()
    answer = "{:0.2f}".format(100 * count_first_class[1]/count_first_class.sum())
    create_answer('3.txt', answer)

def fourth(data):
    """Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров."""
    ages = data['Age']
    answer = "{:0.2f} {:0.2f}".format(ages.mean(), ages.median())
    create_answer('4.txt', answer)

def fifth(data):
    """Коррелируют ли число братьев/сестер/супругов с числом родителей/детей?
    Посчитайте корреляцию Пирсона между признаками SibSp и Parch"""
    corr = data['SibSp'].corr(data['Parch'])
    answer = "{:0.2f}".format(corr)
    create_answer('5.txt', answer)

if __name__ == '__main__':
    data = create_dataFrame()
    first(data)
    second(data)
    third(data)
    fourth(data)
    fifth(data)
