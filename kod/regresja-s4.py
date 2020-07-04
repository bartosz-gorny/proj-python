import matplotlib.pyplot as plt
import numpy as np

 
def oblicz_wartosc(parametry, rok):
    return parametry[0]+parametry[1]*rok+parametry[2]*(rok**2)+parametry[3]*(rok**3)+parametry[4]*(rok**4)
 
def KMNK(Y, x0, x1):
    XT = np.array([x0, x1, x1**2, x1**3,x1**4])
    X = XT.T
 
    XTX = XT@X
    invXTX = np.linalg.inv(XTX)
    XTY=XT@Y
    wspolczynniki = invXTX@XTY
    przewidywane_wartosci_y = []
    for i in tablica_x:
        przewidywane_wartosci_y.append(oblicz_wartosc(wspolczynniki,i))
    return przewidywane_wartosci_y
 
def rysuj(pozycja_rysunku,x0,x1,Y,tablica_x):
    plt.subplot(2,2,pozycja_rysunku)
    plt.plot(x1,Y,'rx')
    plt.plot(tablica_x,KMNK(Y, x0, x1))
    plt.grid(True)
    plt.xlabel('Kolejne lata od 2002')
    plt.ylabel('Liczba wypadków od roku 2002')
    napis='Liczba wypadków w Polsce na podstawie danych z ' + str(np.size(Y)) + 'lat'
    plt.title(napis)
    plt.legend(('dane rzeczywiste','dane modelowane'))
    labels = [2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024]
    plt.xticks(tablica_x, labels, rotation='vertical')
 
prognoza = 25
tablica_x=range(1,prognoza,1)

#Y=liczba wypadków od 2002 roku (sumarycznie)
Y  = np.array([5106083.2, 10213737.6, 15021532.2, 19707156.2, 24662184.2, 29570274.5, 33990047.8, 37873861.7, 41889563, 45596466.5, 49180416.2,
               52678120.9, 55972188.9, 59338919.7, 62617339.9, 65786247.4])
x0 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
x1 = np.array(range(1,17,1))
rysuj(1,x0,x1,Y,tablica_x)

Y  = np.array([5106083.2, 10213737.6, 15021532.2, 19707156.2, 24662184.2, 29570274.5, 33990047.8, 37873861.7, 41889563, 45596466.5, 49180416.2,
               52678120.9])
x0 = np.array([1,1,1,1,1,1,1,1,1,1,1,1])
x1 = np.array(range(1,13,1))
rysuj(2,x0,x1,Y,tablica_x)

Y  = np.array([5106083.2, 10213737.6, 15021532.2, 19707156.2, 24662184.2, 29570274.5, 33990047.8, 37873861.7, 41889563, 45596466.5, 49180416.2,
               52678120.9, 55972188.9])
x0 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1])
x1 = np.array(range(1,14,1))
rysuj(3,x0,x1,Y,tablica_x)

Y  = np.array([5106083.2, 10213737.6, 15021532.2, 19707156.2, 24662184.2, 29570274.5, 33990047.8, 37873861.7, 41889563, 45596466.5, 49180416.2])
x0 = np.array([1,1,1,1,1,1,1,1,1,1,1])
x1 = np.array(range(1,12,1))
rysuj(4,x0,x1,Y,tablica_x)

plt.show()