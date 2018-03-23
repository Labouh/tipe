from statistics import mean
import matplotlib.pyplot as plt

def separe(liste):
    liste_rouge = []
    liste_verte = []
    liste_bleue = []
    for i in liste:
        if i[2] == 1 :
            liste_rouge.append((i[0],i[1]))
        if i[2] == 2 :
            liste_verte.append((i[0],i[1]))
        if i[2] == 3 :
            liste_bleue.append((i[0],i[1]))
    return liste_rouge, liste_verte, liste_bleue

def duree_de_vie_a_t(liste, t):
    aux = []
    i = len(liste)-1
    while i>=0 and t <= liste[i][1] :
        if liste[i][0] <= t  :
            aux.append(liste[i][1] - liste[i][0])
        i -= 1
    if not len(aux) :
        return 0
    else :
        return mean(aux)

def trace_graph(liste,tpsmax):
    T = np.arange(0,tpsmax+1)
    lr,lv,lb = separe(liste)
    yr,yv,yb,ytot = [],[],[],[]
    for t in range(tpsmax+1):
        yr.append(duree_de_vie_a_t(lr,t))
        yv.append(duree_de_vie_a_t(lv,t))
        yb.append(duree_de_vie_a_t(lb,t))
        ytot.append(duree_de_vie_a_t(liste,t))
    plt.plot(T , yr, color='red')
    plt.plot(T , yv,color='green')
    plt.plot(T , yb,color='blue')
    plt.plot(T , ytot,color='gray')
    plt.show()