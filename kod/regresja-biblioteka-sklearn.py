import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


#POBRANIE DANYCH

dataset = pd.read_csv('../dane-csv/lp-rok-lwypsum-00-18.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
dataset1 = pd.read_csv('../dane-csv/lp-rok-lwypsum-00-15.csv')
X1 = dataset1.iloc[:, 1:-1].values
y1 = dataset1.iloc[:, -1].values
dataset2 = pd.read_csv('../dane-csv/lp-rok-lwypsum-00-12.csv')
X2 = dataset2.iloc[:, 1:-1].values
y2 = dataset2.iloc[:, -1].values
dataset3 = pd.read_csv('../dane-csv/lp-rok-lwypsum-00-10.csv')
X3 = dataset3.iloc[:, 1:-1].values
y3 = dataset3.iloc[:, -1].values

#USTALENIE STOPNIA WIELOMIANU
poly_reg = PolynomialFeatures(degree = 2)

#polynomial dla danych 2000-2018
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
#polynomial dla danych 2000-2015
X_poly1 = poly_reg.fit_transform(X1)
lin_reg_21 = LinearRegression()
lin_reg_21.fit(X_poly1, y1)
#polynomial dla danych 2000-2012
X_poly2 = poly_reg.fit_transform(X2)
lin_reg_22 = LinearRegression()
lin_reg_22.fit(X_poly2, y2)
#polynomial dla danych 2000-2010
X_poly3 = poly_reg.fit_transform(X3)
lin_reg_23 = LinearRegression()
lin_reg_23.fit(X_poly3, y3)

#Oś X
tablica_x = [[2000],[2001],[2002],[2003],[2004],[2005],[2006],[2007],[2008],[2009],[2010],[2011],[2012],[2013],[2014],[2015],[2016],[2017],[2018],[2019],[2020],[2021],[2022],[2023],[2024],[2025]]

#FUNKCJA RYSUJĄCA
def rysuj(pozycja_rysunku,x,y,tablica_x,lin_reg):
    plt.subplot(2,2,pozycja_rysunku)
    plt.scatter(x, y, color = 'red')
    plt.plot(tablica_x, lin_reg.predict(poly_reg.fit_transform(tablica_x)), color = 'blue')
    plt.grid(True)
    plt.xlabel('Kolejne lata od 2000')
    plt.ylabel('Liczba wypadków od roku 2000')
    napis='Sumaryczna liczba wypadków w Polsce od 2000r. na podstawie danych z ' + str(np.size(y)) + 'lat'
    plt.title(napis)
    plt.legend(('dane rzeczywiste','dane modelowane'))
    labels = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025]
    plt.xticks(tablica_x, labels, rotation='vertical')



rysuj(1,X,y,tablica_x,lin_reg_2)
rysuj(2,X1,y1,tablica_x,lin_reg_21)
rysuj(3,X2,y2,tablica_x,lin_reg_22)
rysuj(4,X3,y3,tablica_x,lin_reg_23)
plt.show()

#PREDYKCJA ILOŚCI SUMARYCZNYCH WYPADKÓW NA DANY ROK
# print(lin_reg_2.predict(poly_reg.fit_transform([[2030]])))