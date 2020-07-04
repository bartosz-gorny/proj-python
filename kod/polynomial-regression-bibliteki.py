import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('../dane-csv/lp-rok-lwypsum-02-17.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
dataset1 = pd.read_csv('../dane-csv/lp-rok-lwypsum-02-14.csv')
X1 = dataset1.iloc[:, 1:-1].values
y1 = dataset1.iloc[:, -1].values
dataset2 = pd.read_csv('../dane-csv/lp-rok-lwypsum-02-11.csv')
X2 = dataset2.iloc[:, 1:-1].values
y2 = dataset2.iloc[:, -1].values
dataset3 = pd.read_csv('../dane-csv/lp-rok-lwypsum-02-08.csv')
X3 = dataset3.iloc[:, 1:-1].values
y3 = dataset3.iloc[:, -1].values

print(X)
print(y)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


def rysuj(pozycja_rysunku,x,y):
    plt.subplot(2,2,pozycja_rysunku)
    plt.scatter(x, y, color = 'red')
    plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
    plt.grid(True)
    plt.xlabel('Kolejne lata od 2002')
    plt.ylabel('Liczba wypadków od roku 2002')
    napis='Liczba wypadków w Polsce na podstawie danych z ' + str(np.size(y)) + 'lat'
    plt.title(napis)
    plt.legend(('dane rzeczywiste','dane modelowane'))
    labels = [2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024]
    plt.xticks(x, labels, rotation='vertical')

rysuj(1,X,y)
rysuj(2,X1,y1)
rysuj(3,X2,y2)
rysuj(4,X3,y3)
plt.show()

#PREDYKCJA ILOŚCI SUMARYCZNYCH WYPADKÓW NA DANY ROK
print(lin_reg_2.predict(poly_reg.fit_transform([[2030]])))