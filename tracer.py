from statistics import mean
import matplotlib.pyplot as plt
import numpy as np


def separe(liste):
    n = len(liste)
    liste_rouge = np.zeros((n,2),dtype = int)
    liste_verte = np.zeros((n,2),dtype = int)
    liste_bleue = np.zeros((n,2),dtype = int)
    liste_rouge[0] = 1
    liste_verte[0] = 1
    liste_bleue[0] = 1
    for i in liste:
        if i[2] == 1 :
            liste_rouge[liste_rouge[0,0]] = i[0],i[1]
            liste_rouge[0] += 1
        if i[2] == 2 :
            liste_verte[liste_verte[0,0]] = i[0],i[1]
            liste_verte[0] += 1
        if i[2] == 3 :
            liste_bleue[liste_bleue[0,0]] = i[0],i[1]
            liste_bleue[0] += 1
    return liste_rouge[:liste_rouge[0,0]+1], liste_verte[:liste_verte[0,0]+1], liste_bleue[:liste_bleue[0,0]+1]

def duree_de_vie(lr,lv,lb,tmax, save):
    
    datar = np.zeros((len(lr),tmax))
    datav = np.zeros((len(lv),tmax))
    datab = np.zeros((len(lb),tmax))
    datatot = np.zeros((3*len(lr),tmax))
    datar[0],datav[0],datab[0],datatot[0] = 2,2,2,2
    for r in lr :
        dr = r[1] - r[0]
        for p in range(r[0],r[1]):
            datar[datar[0,p],p] = dr
            datar[0,p] += 1
            datatot[datatot[0,p],p] = dr
            datatot[0,p] += 1
    print('rouge done')
    for v in lv :
        dv = v[1] - v[0]
        for p in range(v[0],v[1]):
            datav[datav[0,p],p] = dv
            datav[0,p] += 1
            datatot[datatot[0,p],p] = dv
            datatot[0,p] += 1
    print('verte done')    
    for b in lb :
        db = b[1] - b[0]
        for p in range(b[0],b[1]):
            datab[datab[0,p],p] = db
            datab[0,p] += 1
            datatot[datatot[0,p],p] = db
            datatot[0,p] += 1
    print('bleue done')
    for i in range(tmax):
        if datar[0,i] != 2 :
            datar[1,i] = mean(datar[2:datar[0,i],i])
        if datav[0,i] != 2 :
            datav[1,i] = mean(datav[2:datav[0,i],i])
        if datab[0,i] != 2 :
            datab[1,i] = mean(datab[2:datab[0,i],i])
        if datatot[0,i] != 2 :
            datatot[1,i] = mean(datatot[2:datatot[0,i],i])
    if save:
        print('nom fichier')
        nomfichier = input()
        
        fr =open(nomfichier+'r.txt', 'a')
        fv =open(nomfichier+'v.txt', 'a')
        fb =open(nomfichier+'b.txt', 'a')
        ftot =open(nomfichier+'tot.txt', 'a')
        
        fr.write('datar = ')
        fr.write(str(datar))
        
        fv.write('datav = ')
        fv.write(str(datav))
        
        fb.write('datab = ')
        fb.write(str(datab))
        
        ftot.write('datatot = ')
        ftot.write(str(datatot))
        
        fr.close()
        fv.close()
        fb.close()
        ftot.close()
    
    return datar[1],datav[1],datab[1],datatot[1]

def trace_graph(liste,tmax, save = False):
    T = np.arange(0,tmax)
    lr,lv,lb = separe(liste)
    mr,mv,mb,mtot = duree_de_vie(lr,lv,lb,tmax, save)
    plt.plot(T , mr, color='red')
    plt.plot(T , mv,color='green')
    plt.plot(T , mb,color='blue')
    plt.plot(T , mtot,color='gray')
    plt.show()


def trace_graph_data(datar, darav, datab, datatot, tmax):
    T = np.arange(0,tmax)
    mr,mv,mb,mtot = datar[1],datav[1],datab[1],datatot[1]
    plt.plot(T , mr, color='red')
    plt.plot(T , mv,color='green')
    plt.plot(T , mb,color='blue')
    plt.plot(T , mtot,color='gray')
    plt.show()  