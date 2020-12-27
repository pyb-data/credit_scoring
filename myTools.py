# Librairies d'outils
import os
import datetime
import warnings
import urllib.request

# Librairies mathématiques
import random
import math
import scipy.stats as st
import statsmodels.api as sm

# Librairies de tableaux de données
import numpy as np
import pandas as pd

# Librairies graphiques
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.collections import LineCollection
import seaborn as sns

from sklearn import decomposition
from sklearn import preprocessing


import datetime



def Scatter(X, Y, X_name, Y_name, title="", alpha=0.1, color=None):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10})
    fig = plt.figure(figsize=(8,8))
    if color is None:
        plt.scatter(X, Y, marker = 'o', alpha = alpha)
    else:
        if type(color[0]) != list:
            plt.scatter(X, Y, marker = 'o', alpha = alpha, c=color)
        else:            
            sc = plt.scatter(X, Y, marker = 'o', alpha = alpha, c=color[0])
            plt.legend(sc.legend_elements()[0], np.unique(color[1]))
    plt.xlabel(X_name)
    plt.ylabel(Y_name)
    plt.rcParams.update({'font.size':15})
    if title == "":
        plt.title(X_name + " vs " + Y_name)
    else:
        plt.title(title)
    plt.show()

    pearson = st.pearsonr(X,Y)[0]
    print("Pearson entre {} et {}: {}".format(X_name, Y_name, np.round(pearson,2)))
    print("R2 entre {} et {}: {}".format(X_name, Y_name, np.round(pearson ** 2,2)))



def metrics_model(trainedModel, x_train, x_test, y_train, y_test, record=False, labelModel="", attributes="", dataSet="", comment=""):

    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_squared_log_error
    from sklearn.metrics import r2_score
    ### voir RMSLE
    ### voir RSE, erreur carrée relative qui qui est le complément à 1 du coefficient de détermination R2 (indique une corrélation entre valeurs prédites et vraies valeurs)
    
    exp = False
    
    if exp:
        comment += "EXP"
    
    print("Evaluation sur le jeu d'entrainement")

    start = datetime.datetime.now()
    trainPred = trainedModel.predict(x_train)
    end = datetime.datetime.now()
    duration_train = end - start
    
    if exp:
        pred = np.exp(trainPred)
        real = np.exp(y_train)
    else:
        pred = trainPred
        real = y_train
    

    mae = mean_absolute_error(real, pred)
    print("MAE: {}".format(mae))
        
    R2 = r2_score(real, pred)
    print("R2: {}".format(R2))
    
    mse = mean_squared_error(real, pred)
    rmse = np.sqrt(mse)
    print("RMSE: {}".format(rmse))
    
    pred = [float(np.where(x<0,0,x)) for x in pred]
    rmsle = mean_squared_log_error(real, pred)
    print("RMSLE: {}".format(rmsle))

    
    print("")
    print("Evaluation sur le jeu de test")
    
    start = datetime.datetime.now()
    testPred = trainedModel.predict(x_test)
    end = datetime.datetime.now()
    duration_test = end - start

    if exp:
        pred = np.exp(testPred)
        real = np.exp(y_test)
    else:
        pred = testPred
        real = y_test
        
    mae_test = mean_absolute_error(real, pred)
    print("MAE: {}".format(mae_test))

    R2_test = r2_score(real, pred)
    print("R2: {}".format(R2_test))
    
    mse = mean_squared_error(real, pred)
    rmse_test = np.sqrt(mse)
    print("RMSE: {}".format(rmse_test))
    
    pred = [float(np.where(x<0,0,x)) for x in pred]
    rmsle_test = mean_squared_log_error(real, pred)
    print("RMSLE: {}".format(rmsle_test))

    
    if record:
        iIndex=int(np.where(np.isnan(tableMetrics.index.max()),0, tableMetrics.index.max())) + 1
        tableMetrics.loc[iIndex] = [labelModel, mae, R2, rmse, rmsle, duration_train, mae_test, R2_test, rmse_test, rmsle_test, duration_test, attributes, dataSet, comment]
    


def plot_learning_curves(model, X_train, X_val, y_train, y_val, xscale='', yscale='', legend_loc="upper right"):
    
    from sklearn.metrics import mean_squared_error
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train", color="green" )
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val", color="orange")
    plt.legend(loc=legend_loc, fontsize=14)   # not shown in the book
    plt.xlabel("Training set size", fontsize=14) # not shown
    plt.ylabel("RMSE", fontsize=14)              # not shown
    if xscale.lower() == "log": plt.xscale('log')
    if yscale.lower() == "log": plt.yscale('log')



def exploreRegulLinReg(X, y, param_alpha=[-5, 5, 200], typeReg="ridge", size=1):
    
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso
    
    if typeReg.lower() == "ridge":
        model = Ridge()
    elif typeReg.lower() == "lasso":
        model = Lasso(fit_intercept=False)
    
    alphas = np.logspace(param_alpha[0], param_alpha[1], param_alpha[2]) 
    
    coefs = []
    errors = []
    
    for a in alphas:
        model.set_params(alpha=a)
        #ridge.set_params(alpha=a, solver="cholesky")
        # ridge.set_params(alpha=a, solver="sag")
        model.fit(X, y)
        coefs.append(model.coef_)
        errors.append(np.sqrt(np.mean((model.predict(X) - y) ** 2)))
        
    import matplotlib.pyplot as plt

    
    f = plt.figure(figsize=(15,6))
    plt.rcParams.update({'font.size':10})
    
    #plt.title('titre')
    ax = f.add_subplot(121)

    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    plt.gca().set_xlabel('alpha')
    plt.gca().set_ylabel('weights')    
    plt.xlabel('alpha')
    plt.ylabel('weights')
    ax.set_title("Coefficients de régression {} en fonction de alpha".format(typeReg))
    plt.axis('tight')
    
    ax2 = f.add_subplot(122)    

    ax2.plot(alphas, errors)
    ax2.set_xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('error')
    plt.axis('tight')
    ax2.set_title("RMSE en fonction de alpha")
    
    plt.rcParams.update({'font.size':15})
    plt.suptitle("Recherche du coefficient de régularisation {} optimal (alpha)".format(typeReg))    
    
    plt.show()



def isnull(value, valueIfNull):
    try:
        test=np.isnan(value)
    except Exception:
        return value
    if np.isnan(value):
        return valueIfNull
    else:
        return value
    

# XXX possible de gérer avec une compréhension de liste?
def rename_col(df):
    """Modifies column names of a dataframe: 'ma colonne'>'MaColonne', '23macolonne'>'c23macolonne' """
    for col in df.columns:
        # if the column starts with a with a number, we add a "c" at the beginning
        if col[:1] in ("0","1","2","3","4","5","6","7","8","9"):
            newCol = 'c' 
        else: 
            newCol = '' 
        # if the column name contains spaces, we remove them and we capitalize the first letter of each word
        if ' ' in col:
            for i,elt in enumerate(col.split(" ")): newCol += elt.capitalize() 
        else:
            newCol += col
        newCol = newCol.replace("-","_").replace("(","_").replace(")","_").replace("/","_")
        df.rename(columns={col:newCol}, inplace=True)
        

        
def showAllHist(df, nbCol=6, size=(20,10)):

    quantCol = myDf(df).Stat(returnTypeCol=True)[1]
    nbQuantCol = len(quantCol)

    nbCol = 6
    nbLignes = int(np.where(nbQuantCol / nbCol == int(nbQuantCol / nbCol),nbQuantCol / nbCol ,int(nbQuantCol / nbCol) + 1 ))

    fig = plt.figure(figsize=size)

    i = 1
    for col in quantCol:
        ax = fig.add_subplot(nbLignes,nbCol,i)
        h = ax.hist(df[col], bins=50, color='steelblue', density=True, edgecolor='none')
        ax.set_title(col, fontsize=10)
        i += 1


        
        
def EraseFile(rep, removeRep=False):

    files=os.listdir(rep)
    for i in range(0,len(files)):
        os.remove(rep+'/'+files[i])

    if removeRep:
        os.removedirs(rep) 
        
        
def displayPictures(url, titre=""):
    
    if not os.path.exists("images"):
        os.makedirs("images")
    EraseFile("images")
    
    urllib.request.urlretrieve(url, "images/tmp.jpg")
    nbPlot = 1
    size = 1
    fig = plt.figure(figsize=(size * (18 * nbPlot/2), size * 8))
    
    sub = fig.add_subplot(1,1,1)
    img = mpimg.imread("images/tmp.jpg")
    plt.imshow(img)
    plt.axis('off')
    sub.set_title(titre)
    
    plt.show()

        
def trunc_string(chaine, longueur):
    chaine = str(chaine)
    if len(chaine) < longueur:
        return chaine
    else:
        return chaine[:longueur-1] + ".."
    
def printSingleStat(nb, texte):
    print(str(nb) + " " + texte)


    
def printRatio(nbPart, nbAll, substract=True, text1="", text2=""):
    if substract:
        nb = nbAll-nbPart
    else:
        nb = nbPart
    print(str(nb) + " " + text1 + " " + str(nbAll) + " " + text2 + " (" + str(round(100*(nb)/nbAll,3)) + "%)" )
    
    
def nbCommonWords(string1, string2, lower=True, excludedWords=[]):
    if lower:
        string1 = string1.lower()
        string2 = string2.lower()
    string1 = string1.replace(","," ").replace(";"," ").replace("."," ").replace("-"," ").replace("_"," ").replace("(_)"," ").replace(")"," ")
    string2 = string2.replace(","," ").replace(";"," ").replace("."," ").replace("-"," ").replace("_"," ").replace("(_)"," ").replace(")"," ")
    list1 = string1.split(" ")
    list2 = string2.split(" ")
    for word in excludedWords:
        list1 = removeEltList(list1, word)
        list2 = removeEltList(list2, word)
    return len(commonEltList(list1,list2))


def union(list1,list2):
    liste=[]
    liste.extend(list1)
    liste.extend(list2)
    return liste

    
# retourne les éléments communs à deux listes
def commonEltList(list1, list2):
    return list(set(list1).intersection(list2))

# retourne les éléments d'une liste qui ne sont pas dans une autre
def removeEltList(list1, list2):
    if type(list2) == str:
        list2 = [list2]
    for elt in list2:
        if elt in list1:
            list1.remove(elt)
    return list1
    return list(set(list1)-set(list2)) # on per l'ordre avec ça


def autopctPie(pct,total,thresholdVoid=0, thresholdOnlyPrct=0):
    pct = np.floor(pct * 10) / 10
    if pct < thresholdVoid:
        result = ""
    elif pct < thresholdOnlyPrct:
        result = "{:.1f}%".format(pct)
    else:
        result = "{:.1f}%\n({:d} lignes)".format(pct, int(pct/100.*total))
    return result



def heatMap(table, square=False, annot=False, cmap=sns.diverging_palette(20, 220, n=200), linewidths=.5, colorGradient=[-1,1]):
    
    plt.rcParams.update({'font.size':10})
    if len(colorGradient)==2:
        sns.heatmap(table,square=square,annot=annot, cmap=cmap, linewidths=linewidths, vmin=colorGradient[0], vmax=colorGradient[1])
    else:
        sns.heatmap(table,square=square,annot=annot, cmap=cmap, linewidths=linewidths)
    
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.yticks(rotation=0)
    
    # bug fix, see https://github.com/mwaskom/seaborn/issues/1773
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    
    
    

def eta_squared(x,y):
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT


def test_3D():

    #%matplotlib notebook

    from mpl_toolkits.mplot3d import Axes3D
    
    matplotlib.interactive(True)

    x, y, z = list(np.random.rand(3,10))

    fig = plt.figure( figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z)
    #ax.set_aspect("equal")


def f_test_quiver():
    x = np.linspace(0,1,11)
    y = np.linspace(0,1,11)
    u = [0,0,0.1,0,0,0.05,0,0,0,0,0]
    v = [0,0,0.2,0,0,0.13,0,0,0,0,0]
    plt.quiver(x, y, u, v, scale=1)        
    
    
# Eboulis
def pcaScreePlot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    #plt.show(block=False)

    
# Cercles de projections
def pcaCircles(pca, axis_ranks=[(0,1),(0,2)], labels=None, label_rotation=0, lims=None, size=1):
    
    #%matplotlib inline
    nbPlot=2
    fig = plt.figure(figsize=(size * (18 * nbPlot/2), size * 8))
    numPlot = 1
    plt.rcParams.update({'font.size':10, 'font.style':'normal'})
    
    pcs = pca.components_
    n_comp = len(pcs)

    for d1, d2 in axis_ranks: # affichage des plans factoriels
        if d2 < n_comp:

            # initialisation de la figure
            #fig, ax = plt.subplots(figsize=(7,6))
            sub = fig.add_subplot(1,nbPlot,numPlot)
            numPlot+=1
            plt.rcParams.update({'font.size':13, 'font.style':'italic'})
            sub.set_title("Projection 1")
            plt.rcParams.update({'font.size':10, 'font.style':'normal'})
            
                
            # détermination des paramètres des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                #fonctionne avec:  fig, ax = plt.subplots(figsize=(7,6))
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))

            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)

            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # fixation des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)

            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.rcParams.update({'font.size':18})
    plt.suptitle("Cercle des projections 2D des variables sur les 3 composantes principales")
        
    plt.show()


# Projection des individus sur les composantes principales
def pca2D(X_projected, pca, axis_ranks=[(0,1),(0,2)], labels=None, multivColorCat=None, alpha=1, size=1):

    #%matplotlib inline
    nbPlot=2
    fig = plt.figure(figsize=(size * (18 * nbPlot/2), size * 8))
    numPlot = 1
    plt.rcParams.update({'font.size':10, 'font.style':'normal'})
    
    n_comp = len(pca.components_)
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure
            #fig, ax = plt.subplots(figsize=(7,6))
            sub = fig.add_subplot(1,nbPlot,numPlot)
            numPlot+=1
            plt.rcParams.update({'font.size':13, 'font.style':'italic'})
            sub.set_title("Projection 1")
            plt.rcParams.update({'font.size':10, 'font.style':'normal'})
        
            # affichage des points
            if multivColorCat is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                labelCat, indexCat = np.unique(multivColorCat, return_inverse=True)
                sc = plt.scatter(X_projected[:, d1], X_projected[:, d2], marker = 'o', c = indexCat, alpha = alpha)
                plt.legend(sc.legend_elements()[0], labelCat)

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            
    plt.rcParams.update({'font.size':18})
    plt.suptitle("Cercle des projections 2D des individus sur les 3 composantes principales")            
    plt.rcParams.update({'font.size':10})
    plt.show()

            

    
def pca3D(X_projected, pca, multivColorCat=None):

    #from importlib import reload
    #reload(matplotlib)
    #from matplotlib import pyplot as plt
    
    #import sys
    #sys.modules.pop('matplotlib')
    #import matplotlib.pyplot as plt
    #%matplotlib notebook
    #matplotlib.interactive(True)
    

    from mpl_toolkits.mplot3d import axes3d
    from mpl_toolkits.mplot3d import proj3d
    from matplotlib.patches import FancyArrowPatch

    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
            FancyArrowPatch.draw(self, renderer)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.set_aspect('equal')    # BUG
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    ax.set_zlim(-3,3)
    fig.tight_layout()
    
    x, y, z = X_projected.T
    
    # affichage des points
    if multivColorCat is None:
        ax.scatter(x, y, z, alpha = 0.1, s=1)
    else:
        labelCat, indexCat = np.unique(multivColorCat, return_inverse=True)
        sc = ax.scatter(x, y, z, marker = 'o', c = indexCat, alpha = 0.1)
        plt.legend(sc.legend_elements()[0], labelCat)    
    
    plt.title("PCA 3D")

    # Affichage des vecteurs propres (axes principaux d'inertie)
    pcs = pca.components_
    for i in range(3):
        eigen_vector_x20 = pcs[i] * 20
        eigen_vector = Arrow3D([0, eigen_vector_x20[0]], [0, eigen_vector_x20[1]],[30, 30+eigen_vector_x20[2]], mutation_scale=20, lw=1, arrowstyle="-|>", color="#4c72b0", alpha=.7)
        ax.add_artist(eigen_vector)
        ax.text3D(eigen_vector_x20[0],eigen_vector_x20[1],30+eigen_vector_x20[2],"u{} (x20)".format(i+1), color="#333333")
        
    plt.show()
    

            
            

class mySerie(pd.Series):
    
    def Reverse(self):
        
        if self.name == None:
            name = 0
        else:
            name = self.name
            
        return self.to_frame().reset_index().sort_index(ascending=False).set_index('index')[name]


class myDf(pd.DataFrame):

    neutralColor = '#808080'   # '#7f7f7f'
    barChartColor = '#1f77b4'
    barWidth = 0.4
    
    
    def WhereNotExists(self, df, left_on, right_on):
        dfInner = self.merge(df, left_on=left_on, right_on=right_on)[left_on]
        dfInner["testExists"] = 1
        dfLeft = self.merge(df, left_on=left_on, right_on=right_on, how="left")[left_on]
        dfLeft = dfLeft.merge(dfInner, left_on = left_on, right_on = left_on, how="left")
        dfNotExists = dfLeft[dfLeft.testExists.isnull()][left_on].drop_duplicates()
        return self.merge(dfNotExists, left_on=left_on, right_on=right_on)
    

    def GroupBy(self, keys, aggfields="count", resetIndex=True, *restrictions):
            
        restrictions=list(restrictions)
        
        # keys doit être soit une string (qui les clés d'agrégation séparés par une virgule), soit une liste (qui liste les clés)
        if type(keys) == str:
            keys = [key.strip() for key in keys.split(",")]

        # on prépare le nom des colonnes en sortie (par exemple mean_col, max_col...)
        col_result = []        
        if aggfields != "count":
            if type(aggfields) != dict:
                raise TypeError("Le deuxième paramètre doit être 'count' ou un dictionnaire contenant les champs à agréger avec leurs fonctions d'agrégation.")            
            for elt in aggfields:
                if type(aggfields[elt]) != str:
                    raise TypeError("les fonctions d'agrégation doivent être passée dans une chaine de caractères, séparées par des virgules")
                aggfields[elt]  = [aggfield.strip() for aggfield in aggfields[elt].split(",")]
                for fct_aggr in aggfields[elt]:
                    if fct_aggr not in ("count", "min", "max", "sum", "mean", "nunique"):
                        raise ValueError("La fonction d'agrégation " + fct_aggr + " n'est pas applicable")  
                    col_result.append(fct_aggr + "_" + elt)

        # Vérification que ce sont bien des noms de colonne du dataframe qui sont passés en paramètre
        list_to_check = keys.copy()
        if aggfields != "count":
            list_to_check.extend(list(aggfields.keys()))
        for elt in list_to_check:
            if elt not in self.columns:
                raise ValueError(elt + " n'est pas une colonne du dataframe passé en paramètre")  

        result = self.copy()
        
        # clause where
        for where in restrictions:
            try:
                result = self.query(where)
                restrictions.remove(where)
            except Exception:
                pass # si ça plante ici, ça passera peut être plus loin en clause having

        # on ajoute le count de ligne par clé d'agrégation, on boucle sur les colonnes jusqu'à en trouver une qui ne soit pas dans keys (si elle est dans keys ça plante)
        for col in result.columns:
            try:
                result_count = result.groupby(keys).count()[col].to_frame()
                break
            except Exception:
                pass
        result_count.columns = ["count"]
        
        # group by
        if aggfields != "count":
            result = result.groupby(keys).agg(aggfields)
            result.columns = col_result
            
        
        if aggfields != "count":
            result = result.join(result_count)
            col_result.insert(0,"count")
            result = result[col_result]            
        else:
            result = result_count
        
        
        if resetIndex:
            result = result.reset_index(drop = False)
        
        # clause having
        # 
        for having in restrictions:
            try:
                result = result.query(having)
            except Exception:
                raise ValueError(having + " n'est une clause applicable ni en 'where' ni en 'having'")
                
        return result
   

    
    
    def ColCountLines(self, minCount=0, maxCount=9999999999):
        return self.describe(include="all").transpose().query("(count>=" + str(minCount) + ") & (count<=" + str(maxCount) + ")")["count"] / self.shape[0] * 100
    
    
    def ColCountLinesChart(self, typeChart="bar", displayColName=False, minCount=0, maxCount=9999999999):
        plt.rcParams.update({'font.size':10})
        serieGraph = self.ColCountLines(minCount,maxCount)
        # Selon qu'on affiche le nom des colonnes ou pas
        if not displayColName:
            serieGraph = serieGraph.reset_index(drop=True)
        else:
            serieGraph = mySerie(serieGraph)
        # Dimension de l'axe des colonnes (absisses en bar) selon le nombre de colonnes et selon qu'on affiche les noms de colonne ou pas
        if displayColName:
            x_figsize = int(len(serieGraph)/3)
            if typeChart == "bar":
                x_figsize = np.array([np.array([20,x_figsize]).min(), 8]).max()
            else:
                x_figsize = np.array([np.array([13,x_figsize]).min(), 8]).max()
        else:
            x_figsize = 10
        
        if typeChart == "barh":
            fig = plt.figure(figsize=(8, x_figsize))
            plt.barh(y=serieGraph.index,width=serieGraph,height=myDf.barWidth, color=myDf.neutralColor)
            plt.gca().invert_yaxis()
            #plt.gca().set_xlabel('Nombre de lignes')
            if displayColName:
                pass
                #plt.gca().set_ylabel('Nom des colonnes')
            else:
                plt.gca().set_ylabel('Numéro des colonnes')
        else:
            fig = plt.figure(figsize=(x_figsize, 8))
            plt.bar(x=serieGraph.index,height=serieGraph, width=myDf.barWidth, color=myDf.neutralColor)
            #plt.gca().set_ylabel('Nombre de lignes')
            if displayColName:
                pass
                #plt.gca().set_xlabel('Nom des colonnes')
            else:
                plt.gca().set_xlabel('Numéro des colonnes')     
        
        title = "Nombre de lignes renseignées par colonne"
        if (minCount > 0) and (maxCount == 9999999999):
            title += " (au moins " + str(minCount) + " lignes)" 
        elif (minCount == 0) and (maxCount != 9999999999):
            title += " (au plus " + str(maxCount) + " lignes)" 
        elif (minCount > 0) and (maxCount != 9999999999):
            title += " (entre " + str(minCount) + " et " + str(maxCount) + " lignes)" 
        plt.gca().set_title(title)
        
        
        if displayColName:
            plt.xticks(rotation=45, ha="right", rotation_mode="anchor")

    
    def WeightMean(self, col, colWeight):
        sumWeight = self[colWeight].sum()
        sumWeightCol = self.apply(lambda x: x[col] * x[colWeight], axis=1).sum()
        return sumWeightCol / sumWeight
    
    
        
    def ClassifyFeature(self,feature, typeFeature="tbd", thresholdContinu=20):
        
        if typeFeature == "tbd":
            dTypeFeature = self.dtypes[feature]
            if dTypeFeature == 'float64': # Si la colonne est float alors que toutes les valeurs sont entières, alors on convertit en entier (pour éviter d'avoir "2.0" dans les libellé et pour identifier une statistique ordinale)        
                typeFeature = 'continue'
                if len(self[feature].unique()) <= thresholdContinu:
                    typeFeature = 'discrete'
                    for index, row in self[[feature]].dropna().drop_duplicates().iterrows():
                        if np.floor(row[feature]) != row[feature]:
                            typeFeature = 'continue'
            elif dTypeFeature== int:
                if len(self[feature].unique()) <= thresholdContinu:
                    typeFeature = 'discrete'
                else:
                    typeFeature = 'continue'
            else:
                # Si les libellés des catégories sont des lettres, on considère qu'on est en ordinal (et par défaut on fait un bar chart)
                if self[[feature]].dropna().apply(lambda x: len(str(x[feature])), axis=1).max() == 1:
                    typeFeature = "ordinale"
                else:
                    typeFeature = "nominale"    

        return pd.DataFrame({"feature": [feature], "typeFeature": [typeFeature]})
                
                

            
        
    def StatUniv(self, col="all", size=1, statsNan=False, pltTitle="tbd", typeChart="tbd", typeFeature="tbd", threshold=2.5,  modalities=10.1, 
                 lower=False, truncateLabel=35, histDensity=False, histxLog=False, histyLog=False, histBins=100, pieColors="default", 
                 pieStartAngle=0.001, pieThresholdVoidLabel=2, pieThresholdOnlyPrctLabel=10, pieThresholdVoidModality=1, violinPlot=False, boxShowfliers=True, thresholdContinu=20):

        barh = False
        if typeChart == "barh":
            typeChart = "bar"
            barh = True
           
        # On garde en mémoire si certains champs ont été forcés
        forcedTypeChart = (typeChart != "tbd")
        
        # La série de données à représenter graphiquement
        serieGraph = self[[col]].dropna()
        
        # On passe les noms des modalités en minuscule afin d'éviter les doublons dûs à la casse
        if lower & (serieGraph.dtypes[col] == 'object'):
            serieGraph = serieGraph.apply(lambda x: str(x[col]).strip().lower(), axis=1).to_frame()
            serieGraph.columns = [col]      
            
        # Détermination du type de graph en fonction du type de variable
        if typeChart == "tbd":
            if typeFeature == "nominale":
                typeChart = "pie"
            elif (typeFeature == "ordinale") | (typeFeature == 'discrete'):
                if len(serieGraph[col].unique()) > 3:
                    typeChart = "bar"
                else:
                    typeChart = "pie"
            elif typeFeature == "continue":
                typeChart = "hist"        
        
        
        ############################################################################################
        # Préparation de la série statistique des valeurs non manquantes à représenter graphiquement
        ############################################################################################
        if typeChart == 'hist':
            serieGraphHist = np.array(serieGraph[col])
            
        # Liste des données à mettre en graphique avec pour chaque modalité le nombre de ligne
        serieGraph = serieGraph[col].value_counts()
        nbLigneTotal = sum(serieGraph)
        serieGraph = pd.DataFrame({col:np.array(serieGraph.index), "NbLigne":serieGraph})   
    

                
        ###############################################################
        # Regroupement des petities modalités dans une modalité "autres"
        ###############################################################

        # Dans le cas d'une variable nominale (catégorielle), on gère une modalité "autres" qui regroupe les modalités les moins représentées
        nbBarAutres = 0
        nbLignesAutres = 0
        if (typeFeature == 'nominale') | (typeChart == "pie"):
            # Gestion du threshold qui va servir à regrouper des modalités dans une modalité "autres"
            if type(threshold)==str:
                threshold=threshold.replace(",",".")
                if threshold[-1] != "%":
                    warnings.warn("Si vous passez le threshold en string, écrivez un nombre suivi de '%'. Le threshold est mis à 2,5%.")
                    threshold = 2.5
                else:
                    try:
                        threshold = float(threshold[0:-1])
                    except Exception:
                        warnings.warn(warningThreshold)
                        threshold = 2
            elif (type(threshold)==int) | (type(threshold)==float):
                if (threshold > 0) & (threshold < 1):
                    warnings.warn("Le threshold indiqué semble inférieur à 1%.")
                if (threshold > 100) | (threshold < 0):
                    warnings.warn("Le threshold est un pourcentage et ne doit pas dépasser 100 ou être inférieur à 0. Le threshold est mis à 2,5%.")
                    threshold = 2.5

            # Pour éviter d'avoir trop de modalités représentées dans le graphique, on limite le nombre selon deux critères passés en paramètre:
            #      - threshold (le nombre de ligne que la modalité doit au minimum avoir, en proportion du nombre total de lignes pour lesquelles la modalité est renseignée)
            #      - modalities (nombre de modalités à afficher au maximum)
            # Les lignes qui ne vérifient pas ces conditions sont regroupées dans une ou plusieurs catégories "autre"

            # Si threshold est passé en paramètre et pas modalities, alors on calcule modalities pour qu'il soit en phase avec threshold
            if (threshold != 2.5) & (modalities == 10.1):
                modalities = len(serieGraph.query("NbLigne >= " + str(threshold/100.*nbLigneTotal))) 
            else:
                modalities=int(modalities)
                threshold = 999
            nbLignesAutres = 0  # nombre de lignes des autres modalités
            nbAutresModalites = len(serieGraph) - modalities   # nombre des autres modalités
            # S'il n'y a qu'une autre modalité, on ne la remplace pas une modalité "autres", il faut au moins deux autres modalités pour regrouper
            if nbAutresModalites > 1:
                serieGraphAutres = serieGraph.tail(nbAutresModalites)
                nbLignesAutres = serieGraphAutres.NbLigne.sum()
                maxAutres = serieGraphAutres.head(1).NbLigne.values[0]
                serieGraph = serieGraph.head(modalities)
                # Si la distribution ne contient que des petites modalités, alors la catégorie "autres" prend trop de place et un pie chart n'est plus lisible
                # dans ce cas on passe en bar chart
                if nbLignesAutres / nbLigneTotal > 0.95 and not forcedTypeChart:
                    typeChart = "bar"
                if typeChart=="pie":
                    # Dans le cas d'un pie chart, on regroupe toutes les autres modalités dans une seule portion du pie chart
                    if threshold == 999:
                        # le threshold affiché est le NbLigne (en %) de la modalité la moins courante qui n'est pas dans le groupe "autres"
                        threshold = 100 * maxAutres / nbLigneTotal
                        # on tronque à un chiffre ou plus si il faut) après la virgule
                        if threshold > 0.1:
                            threshold = np.floor(threshold * 10) / 10
                        elif threshold > 0.01:
                            threshold = np.floor(threshold * 100) / 100
                        elif threshold > 0.001:
                            threshold = np.floor(threshold * 1000) / 1000
                        elif threshold > 0.0001:
                            threshold = np.floor(threshold * 10000) / 10000
                        elif threshold > 0.00001:
                            threshold = np.floor(threshold * 100000) / 100000
                        elif threshold > 0.000001:
                            threshold = np.floor(threshold * 1000000) / 1000000
                        else:
                            threshold = 0
                    lblAutre = str(nbAutresModalites) + " autres modalités (<= " + str(threshold) + "%)"
                    serieGraphAutres = pd.DataFrame([[lblAutre,nbLignesAutres]], columns = serieGraph.columns)
                elif typeChart == "bar":
                    # Dans le cas d'un bar chart, on va regrouper les autres modalités dans plusieurs "autres barres"
                    nbBarAutres = 0
                    serieGraphAutres = serieGraphAutres.NbLigne.value_counts().reset_index()
                    serieGraphAutres.columns = ['NbLigne','NbModalities']
                    serieGraphAutres = serieGraphAutres.sort_values("NbLigne", ascending=False)
                    serieGraphAutres = serieGraphAutres.reset_index(drop=True)

                    NbModalities = serieGraphAutres.query("NbLigne == 1").NbModalities.values[0]
                    if NbModalities >= nbAutresModalites / 10:
                        # Si le nombre de modalités à une ligne est significatif, on créé une barre à une ligne
                        nbBarAutres += 1
                        lblAutre = str(NbModalities) + " autres modalités à 1 ligne"
                        serieGraphAutres2 = pd.DataFrame([[lblAutre,1]], columns = serieGraph.columns)
                        serieGraphAutres = serieGraphAutres.drop(serieGraphAutres[serieGraphAutres.NbLigne==1].index)
                        # Ainsi qu'une barre à deux lignes
                        NbModalities = 0
                        try:
                            NbModalities = serieGraphAutres.query("NbLigne == 2").NbModalities.values[0]
                        except Exception:
                            pass
                        if NbModalities > 0:
                            nbBarAutres += 1
                            lblAutre = str(NbModalities) + " autres modalités à 2 lignes"
                            serieGraphAutres2 = serieGraphAutres2.append(pd.DataFrame([[lblAutre,2]], columns = serieGraph.columns))
                            serieGraphAutres = serieGraphAutres.drop(serieGraphAutres[serieGraphAutres.NbLigne==2].index)
                            # et une barre à trois lignes
                            if len(serieGraphAutres.query("NbLigne == 3")) > 0:
                                NbModalities = serieGraphAutres.query("NbLigne == 3").NbModalities.values[0]
                                if NbModalities > 0:
                                    nbBarAutres += 1
                                    lblAutre = str(NbModalities) + " autres modalités à 3 lignes"
                                    serieGraphAutres2 = serieGraphAutres2.append(pd.DataFrame([[lblAutre,3]], columns = serieGraph.columns))
                                    serieGraphAutres = serieGraphAutres.drop(serieGraphAutres[serieGraphAutres.NbLigne==3].index)

                    if len(serieGraphAutres) > 1000 and (serieGraphAutres.NbLigne.max() / serieGraphAutres.NbLigne.min() > 10):
                        nbBarAutres += 1
                        nbHalfAutre = int(np.floor(len(serieGraphAutres) / 2))
                        moyLignesAutre = int(np.floor(myDf(serieGraphAutres.tail(nbHalfAutre)).WeightMean("NbLigne","NbModalities")))
                        nbAutresModalites = serieGraphAutres.tail(nbHalfAutre).NbModalities.sum()
                        lblAutre = str(nbAutresModalites) + " autres modalités à " + str(moyLignesAutre) + " lignes en moy"
                        serieGraphAutres3 = pd.DataFrame([[lblAutre,moyLignesAutre]], columns = serieGraph.columns)
                        serieGraphAutres = serieGraphAutres.head(len(serieGraphAutres) - nbHalfAutre)

                    if len(serieGraphAutres) > 100 and (serieGraphAutres.NbLigne.max() / serieGraphAutres.NbLigne.min() > 10):
                        nbBarAutres += 1
                        nbHalfAutre = int(np.floor(len(serieGraphAutres) / 2))
                        moyLignesAutre = int(np.floor(myDf(serieGraphAutres.tail(nbHalfAutre)).WeightMean("NbLigne","NbModalities")))
                        nbAutresModalites = serieGraphAutres.tail(nbHalfAutre).NbModalities.sum()
                        lblAutre = str(nbAutresModalites) + " autres modalités à " + str(moyLignesAutre) + " lignes en moy"
                        serieGraphAutres4 = pd.DataFrame([[lblAutre,moyLignesAutre]], columns = serieGraph.columns)
                        serieGraphAutres = serieGraphAutres.head(len(serieGraphAutres) - nbHalfAutre)

                    if len(serieGraphAutres) > 30 and (serieGraphAutres.NbLigne.max() / serieGraphAutres.NbLigne.min() > 10):
                        nbBarAutres += 1
                        nbHalfAutre = int(np.floor(len(serieGraphAutres) / 2))
                        moyLignesAutre = int(np.floor(myDf(serieGraphAutres.tail(nbHalfAutre)).WeightMean("NbLigne","NbModalities")))
                        nbAutresModalites = serieGraphAutres.tail(nbHalfAutre).NbModalities.sum()
                        lblAutre = str(nbAutresModalites) + " autres modalités à " + str(moyLignesAutre) + " lignes en moy"
                        serieGraphAutres5 = pd.DataFrame([[lblAutre,moyLignesAutre]], columns = serieGraph.columns)
                        serieGraphAutres = serieGraphAutres.head(len(serieGraphAutres) - nbHalfAutre)

                    if len(serieGraphAutres) > 0:
                        nbBarAutres += 1
                        moyLignesAutre = int(np.floor(myDf(serieGraphAutres).WeightMean("NbLigne","NbModalities")))
                        nbAutresModalites = serieGraphAutres.NbModalities.sum()
                        lblAutre = str(nbAutresModalites) + " autres modalités à " + str(moyLignesAutre) + " lignes en moy"
                        serieGraphAutres = pd.DataFrame([[lblAutre,moyLignesAutre]], columns = serieGraph.columns)
                    try:
                        serieGraphAutres = serieGraphAutres.append(serieGraphAutres2)
                    except:
                        pass
                    try:
                        serieGraphAutres = serieGraphAutres.append(serieGraphAutres3)
                    except:
                        pass
                    try:
                        serieGraphAutres = serieGraphAutres.append(serieGraphAutres4)
                    except:
                        pass
                    try:
                        serieGraphAutres = serieGraphAutres.append(serieGraphAutres5)
                    except:
                        pass

                    serieGraphAutres = serieGraphAutres.sort_values("NbLigne", ascending=False)


                serieGraph = serieGraph.append(serieGraphAutres)  

        
        ###################
        # Mise en graphique
        ###################
        # Détermination du nombre de graphiques
        nbPlot = 0
        if statsNan:
            nbPlot += 1 # pour un pie des NaNs
        nbPlot += 1  # pour le graphique principal
        if (typeFeature == "continue") | (typeChart == "hist"):
            nbPlot += 1 # Pour un boxplot
        
        # variable pour les 1 à 4 graphiques
        numPlot = 0
        
        fig = plt.figure(figsize=(size * (18 * nbPlot/2), size * 8))
        
        # S'il n'y a pas de données, on force la représentation des valeurs manquantes et on ne représente pas graphiquement les données
        noData = False
        if len(serieGraph) == 0:
            statsNan=True
            noData = True
            nbPlot = 1
            
        
        # 1 - Affichage d'un pie chart qui représente les valeurs manquantes
        if statsNan:
            numPlot += 1
            sub = fig.add_subplot(1,nbPlot,numPlot)
            plt.rcParams.update({'font.size':13, 'font.style':'italic'})
            sub.set_title("Données manquantes vs renseignées")
            plt.rcParams.update({'font.size':10, 'font.style':'normal'})
            labels = ["Renseignées (" + str(len(self[col].dropna().unique())) + " modalités)", "Manquantes"]
            renseigne = len(self[self[col].notnull()]) 
            nonRenseigne = len(self[self[col].isna()]) 
            sizes = [renseigne, nonRenseigne]
            #plt.pie(sizes, labels=labels, autopct=lambda pct: autopctPie(pct,renseigne+nonRenseigne), colors = ['#2ca02c', '#d62728'],startangle=90-renseigne/ len(self)*360/2 + 0.0001)
            #plt.pie(sizes, labels=labels, autopct=lambda pct: "{:.1f}%\n({:d} lignes)".format(np.floor(pct * 10) / 10, int(pct/100.*(renseigne+nonRenseigne))), colors = ['#2ca02c', '#d62728'],startangle=90-renseigne/ len(self)*360/2 + 0.0001)            
            plt.pie(sizes, labels=labels, autopct=lambda pct: "{:.1f}%\n({:d} lignes)".format(round(pct * 10) / 10, int(pct/100.*(renseigne+nonRenseigne))), colors = ['#2ca02c', '#d62728'],startangle=90-renseigne/ len(self)*360/2 + 0.0001)            
            centre_circle = plt.Circle((0,0),0.70,fc='white')
            #fig = plt.gcf()
            #fig.gca().add_artist(centre_circle)
            plt.gca().add_artist(centre_circle)
        
        if not noData:

            # 2 - Affichage du graphique principal
            numPlot += 1
            sub = fig.add_subplot(1,nbPlot,numPlot)
            plt.rcParams.update({'font.size':13, 'font.style':'italic'})
            sub.set_title("Distribution des données renseignées")
            plt.rcParams.update({'font.size':10, 'font.style':'normal'})

            if typeChart == "hist":
                serieGraph = serieGraphHist
                plt.hist(serieGraphHist, density=histDensity, bins=histBins)
                plt.xlabel(col)
                if histxLog:
                    plt.xscale('log')
                if histyLog:
                    plt.yscale('log')
                if histDensity == True:
                    plt.ylabel('Fréquence')
                else:
                    plt.ylabel('Nombre de lignes')
                    
            if typeChart=="bar":
                # Si la variable est ordinale, on ordonne selon la modalité (en prenant soin de garder la modalité "autres" à la fin)
                if (typeFeature == "ordinale") | (typeFeature == "discrete"):
                    serieGraph = serieGraph.sort_values(col)
                #serieGraph[col] = serieGraph.apply(lambda x: trunc_string(x[col],truncateLabel), axis=1)
                barColors = [self.barChartColor for _ in np.arange(modalities)]
                if nbBarAutres > 0:
                    barColors.extend([self.neutralColor for _ in np.arange(nbBarAutres)])
                #return serieGraph
                #serieGraph[col] = serieGraph.apply(lambda x: trunc_string(x[col],int(truncateLabel*1.5)), axis=1)
                #plt.bar(x=serieGraph.apply(lambda x: trunc_string(x[col],truncateLabel*1.5), axis=1), height=serieGraph.NbLigne, color=barColors)

                if barh:
                    plt.barh(y=serieGraph.apply(lambda x: str(x[col]), axis=1), width=serieGraph.NbLigne, color=barColors)
                    plt.gca().invert_yaxis()
                else:
                    plt.bar(x=serieGraph.apply(lambda x: str(x[col]), axis=1), height=serieGraph.NbLigne, color=barColors)
                plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
                plt.gca().set_xlabel(col)
                plt.gca().set_ylabel('Nombre de lignes')
                #plt.yscale('log')
                #plt.set_yticklabels(['1e%i' % np.round(np.log(y)/np.log(10)) for y in y_labels])

            if typeChart=="pie":
                
                # On va modifier les données pour l'affichage, on créé une copie (car les données originale vont servir éventuellement plus loin pour le boxplot)
                serieGraphCopy = serieGraph.copy()
                serieGraphCopy[col] = serieGraphCopy.apply(lambda x: trunc_string(x[col],truncateLabel), axis=1)
                serieGraphCopy.at[serieGraphCopy.NbLigne < pieThresholdVoidModality / 100 * nbLigneTotal,[col]] = ""
                labels = serieGraphCopy[col]
                sizes = serieGraphCopy.NbLigne
                # Gestion des couleurs
                if pieColors == "default":  ##0110d0  1f77b4
                    listColors = ['#1f77b4', '#d0c101', '#a83c09']
                    for i in range(10):
                        listColors.extend(['#fc5a50', '#8c564b',   '#9467bd',    '#cf6275', '#e377c2'])
                elif pieColors == "default_plt":# '#380282',
                    listColors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                else:
                    warnings.warn("La couleur passée en paramètre n'est pas 'default' ou 'default_plt', les couleurs seront donc choisies au hasard")
                    listColors=[]
                for i in np.arange(10000):
                    # Au cas où on voudrait mettre mettre 10000 modalités dans un pie, il faut une couleur pour chacune
                    # Et si l'utilisateur saisi autre chose que "default" ou "default_plt", alors toutes les couleurs seront choisies au hasard
                    listColors.append([random.random(),random.random(),random.random()])
                listColors = listColors[0:len(sizes)]
                if nbLignesAutres > 0:
                    listColors = listColors[0:len(sizes)-1]
                    # On affiche "autres" en gris
                    listColors.append(myDf.neutralColor)
                # Setup du pie chart
                #plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=startAngle)
                total = sum(sizes)
                if pieStartAngle == 0.001:
                    nbLigneLastLine = serieGraphCopy.tail(1).NbLigne.values[0]
                    nbLignePrevLastLine = serieGraphCopy.tail(2).head(1).NbLigne.values[0]
                    if nbLigneLastLine > nbLignePrevLastLine:
                        # On est dans le cas où "autres" regroupe beaucoup de lignes ("autre" est dans la dernière ligne de serieGraph)
                        # On a donc beaucoup de modalités avec un petit nombre de lignes
                        # Pour mieux les lire sur le pie chart, on va ramener le plus petite ligne juste au dessus de l'angle 0%
                        # Pour cela, il sufit de remonter la part de camembert "autre" qui se termine à 0%
                        pieStartAngle = 360 * (nbLigneLastLine + nbLignePrevLastLine/2) / nbLigneTotal
                    pieStartAngle = 0 # plus simple comme ça...

                plt.pie(sizes, labels=labels, autopct=lambda pct: autopctPie(pct,total,pieThresholdVoidLabel,pieThresholdOnlyPrctLabel), colors=listColors, startangle=pieStartAngle)

                
            if typeChart == "hist":
  
                mean = np.mean(serieGraph)
                std = np.std(serieGraph)
                mini = np.min(serieGraph)
                maxi = np.max(serieGraph)
                q25 = np.quantile(serieGraph, 0.25)
                q75 = np.quantile(serieGraph, 0.75)
                median = np.median(serieGraph)
                skewness = st.skew(serieGraph)
                kurtosis = st.kurtosis(serieGraph)

                numPlot += 1
                sub = fig.add_subplot(1,nbPlot,numPlot)

                if violinPlot:
                    plt.rcParams.update({'font.size':13, 'font.style':'italic'})
                    sub.set_title("max, min, médiane, quartiles, moyenne")
                    plt.rcParams.update({'font.size':10, 'font.style':'normal'})
                    vp = plt.violinplot(serieGraph, positions=[0],showmedians = False, showmeans = False)
                    color = vp['cbars'].get_color()[0]
                    # plot the IQR, median and mean
                    xx = np.array([-1,1])*0.075 - 0.005
                    plt.fill_between(xx,*np.percentile(serieGraph,[25,75]),color=color,alpha=0.5)
                    #plt.plot(xx,[np.median(serieGraph)]*2,'-',color=color)
                    #plt.plot(xx,[np.mean(serieGraph)]*2,'-',color=self.neutralColor)                    
                    plt.plot(xx,[median]*2,'-',color=color)
                    plt.plot(xx,[mean]*2,'-',color=self.neutralColor)
                    # plot the legend
                    plt.plot([],[],color=self.neutralColor,label='mean')
                    plt.plot([],[],color=color,label='median')
                    plt.fill_between([],[],color=color,label='IQR')
                    if histxLog:
                        plt.yscale('log')
                    if histyLog:
                        plt.xscale('log') 
                    plt.legend()
                else:
                    plt.rcParams.update({'font.size':13, 'font.style':'italic'})
                    sub.set_title("max, min, médiane, quartiles")
                    plt.rcParams.update({'font.size':10, 'font.style':'normal'})
                    sns.boxplot(y=serieGraph,showfliers=boxShowfliers)
                    if histxLog:
                        plt.yscale('log')
                    if histyLog:
                        plt.xscale('log')                    
            
                
            #else:
            #    numPlot += 1
            #    sub = fig.add_subplot(1,nbPlot,numPlot)
            #    serieGraph = np.array(serieGraph_bx.NbLigne)
            #    vp = plt.violinplot(serieGraph, positions=[0],showmedians = False, showmeans = False)   
            #    plt.gca().set_xlabel('Nombre de modalités')
            #    plt.gca().set_ylabel('Nombre de lignes')
            #    plt.rcParams.update({'font.size':13, 'font.style':'italic'})
            #    if serieGraph_bx.NbLigne.min() < 100 * serieGraph_bx.NbLigne.max():
            #        plt.yscale('log')
            #        sub.set_title("Distribution des counts (éch. log.)")
            #    else:
            #        sub.set_title("Distribution des données renseignées")
        
        plt.rcParams.update({'font.size':18})
        if pltTitle == "tbd":
            plt.suptitle("Statistique univariée du champ " + col)
        else:
            plt.suptitle(pltTitle)
        plt.rcParams.update({'font.size':10, 'font.style':'normal'})
            
        plt.show()

        if typeFeature=="continue":
            #plt.axis("off")
            #plt.text(0,1,"Mesures de position")
            print("Mesures de position")
            print(" - Moyenne: " + str(round(mean,2)))
            print(" - Médiane: " + str(round(median,2)))
            print("Mesures de dispertion")
            print(" - Ecart type: " + str(round(std,2)))
            print(" - Min: " + str(round(mini,2)))
            print(" - Q25: " + str(round(q25,2)))
            print(" - Q75: " + str(round(q75,2)))
            print(" - Max: " + str(round(maxi,2)))
            print("Mesures de forme")
            print(" - Skewness (asymétrie): " + str(round(skewness,2)))      
            print(" - Kurtosis (applatissement): " + str(round(kurtosis,2)))
        
        
        
        
    def StatBivQuantQuant(self, features, size=1, statsNan=True, pltTitle="tbd", typeChart="tbd", typeFeature="tbd", threshold=2.5,  modalities=10.1, lower=False, truncateLabel=35, histDensity=True, histxLog=False, histyLog=False, histBins=100, pieColors="default", pieStartAngle=0.001, pieThresholdVoidLabel=2, pieThresholdOnlyPrctLabel=10, pieThresholdVoidModality=1,violinPlot=False, boxShowfliers=True,thresholdContinu=20):

        
        # Détermination du nombre de graphiques
        nbPlot = 3
        
        # variable pour les 1 à 4 graphiques
        numPlot = 0
        
        fig = plt.figure(figsize=(size * (18 * nbPlot/2), size * 8))
        
        
        seriesGraph = self[[features[0],features[1]]].dropna()
        
        #print(features)
        #print(seriesGraph)
        #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx")
            
        # 1 - Affichage d'un scatter plot
        numPlot += 1
        sub = fig.add_subplot(1,nbPlot,numPlot)
        plt.rcParams.update({'font.size':13, 'font.style':'italic'})
        sub.set_title(features[0] + " et " + features[1])
        plt.rcParams.update({'font.size':10, 'font.style':'normal'})
        Y = seriesGraph[features[1]]
        X = seriesGraph[[features[0]]]
        X = X.copy() # On modifiera X, on en crée donc une copie
        X['intercept'] = 1.
        result = sm.OLS(Y, X).fit() # OLS = Ordinary Least Square (Moindres Carrés Ordinaire)
        a,b = result.params[features[0]],result.params['intercept']
        a3 = a
        b3 = b

        plt.scatter(seriesGraph[features[0]],seriesGraph[features[1]], alpha=0.2)
        tmp = np.linspace(seriesGraph[features[0]].min(), seriesGraph[features[0]].max(), 1000)
        plt.plot(tmp,[a*x+b for x in tmp], color='red')
        plt.gca().set_xlabel(features[0])
        plt.gca().set_ylabel(features[1])


        # 2 - Affichage de boites parallèles
        numPlot += 1
        sub = fig.add_subplot(1,nbPlot,numPlot)
        plt.rcParams.update({'font.size':13, 'font.style':'italic'})
        sub.set_title("Distribution de " + features[1] + " par rapport à " + features[0])
        plt.rcParams.update({'font.size':10, 'font.style':'normal'})

        taille_classe = (seriesGraph[features[0]].max() - seriesGraph[features[0]].min()) / 7 # taille des classes pour la discrétisation
        
        # Propriétés graphiques
        medianprops = {'color':"black"}
        meanprops = {'marker':'o', 'markeredgecolor':'black',
                        'markerfacecolor':'firebrick'}

        groupes = [] # va recevoir les données agrégées à afficher

        # on calcule des tranches allant du minimum au maximum par paliers de taille taille_classe
        tranches = np.arange(seriesGraph[features[0]].min(), seriesGraph[features[0]].max(), taille_classe)
        tranches += taille_classe/2 # on décale les tranches d'une demi taille de classe
        #tranches = tranches.round(2)
        indices = np.digitize(seriesGraph[features[0]], tranches) # associe chaque valeur à son numéro de classe

        for ind, tr in enumerate(tranches): # pour chaque tranche, ind reçoit le numéro de tranche et tr la tranche en question
            data = seriesGraph.loc[indices==ind,features[1]] # sélection des lignes de la tranche ind
            if len(data) > 0:
                g = {
                    'valeurs': data,
                    'centre_classe': tr-(taille_classe/2),
                    'taille': len(data),
                    'quartiles': [np.percentile(data,p) for p in [25,50,75]]
                }
                groupes.append(g)

        # affichage des boxplots
        plt.boxplot([g["valeurs"] for g in groupes],
                    positions= [g["centre_classe"] for g in groupes], # abscisses des boxplots
                    showfliers= boxShowfliers, # prise en compte des outliers
                    widths= taille_classe*0.7, # largeur graphique des boxplots
                    showmeans=True, meanprops=meanprops, medianprops=medianprops, patch_artist=True
        )
        plt.xticks(rotation=90)

        for g in groupes:
            plt.text(g["centre_classe"],0,"(n={})".format(g["taille"]),horizontalalignment='center',verticalalignment='top')     
        plt.gca().set_xlabel(features[0])
        plt.gca().set_ylabel(features[1])



        # 3 - Affichage de boites parallèles
        numPlot += 1
        sub = fig.add_subplot(1,nbPlot,numPlot)
        plt.rcParams.update({'font.size':13, 'font.style':'italic'})
        sub.set_title("Distribution de " + features[0] + " par rapport à " + features[1])
        plt.rcParams.update({'font.size':10, 'font.style':'normal'})

        taille_classe = (seriesGraph[features[1]].max() - seriesGraph[features[1]].min()) / 7 # taille des classes pour la discrétisation

        groupes = [] # va recevoir les données agrégées à afficher

        # on calcule des tranches allant de 0 au solde maximum par paliers de taille taille_classe
        tranches = np.arange(seriesGraph[features[1]].min(), seriesGraph[features[1]].max(), taille_classe)
        tranches += taille_classe/2 # on décale les tranches d'une demi taille de classe
        #tranches = tranches.round(2)
        indices = np.digitize(seriesGraph[features[1]], tranches) # associe chaque solde à son numéro de classe

        for ind, tr in enumerate(tranches): # pour chaque tranche, ind reçoit le numéro de tranche et tr la tranche en question
            data = seriesGraph.loc[indices==ind,features[0]] # sélection des individus de la tranche ind
            if len(data) > 0:
                g = {
                    'valeurs': data,
                    'centre_classe': tr-(taille_classe/2),
                    'taille': len(data),
                    'quartiles': [np.percentile(data,p) for p in [25,50,75]]
                }
                groupes.append(g)

        # affichage des boxplots
        plt.boxplot([g["valeurs"] for g in groupes],
                    positions= [g["centre_classe"] for g in groupes], # abscisses des boxplots
                    showfliers= boxShowfliers, # prise en compte des outliers
                    widths= taille_classe*0.7, # largeur graphique des boxplots
                    vert = False,
                    showmeans=True, meanprops=meanprops, medianprops=medianprops, patch_artist=True

        )
        plt.xticks(rotation=90)

        for g in groupes:
            plt.text(0,g["centre_classe"],"(n={})".format(g["taille"]),horizontalalignment='center',verticalalignment='top')     
        plt.gca().set_xlabel(features[0])
        plt.gca().set_ylabel(features[1])


        plt.rcParams.update({'font.size':18})
        if pltTitle == "tbd":
            plt.suptitle("Statistique bivariée des champs " + features[0] + " et " + features[1])
        else:
            plt.suptitle(pltTitle)
        plt.rcParams.update({'font.size':10, 'font.style':'normal'})

        plt.show()


        # Divers prints
        tmp = st.pearsonr(seriesGraph[features[0]],seriesGraph[features[1]])[0] # coeff de correlation linéaire
        print("Covariance entre " + features[0] + " et "  + features[1] + " : {} ".format(np.cov(seriesGraph[features[0]],seriesGraph[features[1]],ddof=0)[1,0]))
        print("Covariance débiaisée entre " + features[0] + " et "  + features[1] + " : {} ".format(np.cov(seriesGraph[features[0]],seriesGraph[features[1]],ddof=1)[1,0]))
        print("Coefficient de correlation linéaire entre " + features[0] + " et "  + features[1] + " : {} ".format(tmp))
        print("Régression linéaire entre " + features[0] + " et "  + features[1] + ": y = {}x + {}".format(a, b))
        print("Coefficient de détermination entre " + features[0] + " et "  + features[1] + " : {} ".format(tmp * tmp))


    def StatBivQualQuant(self, features, size=1, statsNan=True, pltTitle="tbd", typeChart="tbd", typeFeature="tbd", threshold=2.5,  modalities=10.1, lower=False, truncateLabel=35, histDensity=True, histxLog=False, 
                         histyLog=False, histBins=100, pieColors="default", pieStartAngle=0.001, pieThresholdVoidLabel=2, pieThresholdOnlyPrctLabel=10, pieThresholdVoidModality=1,violinPlot=False, boxShowfliers=False, 
                         returnFig=False,
                        thresholdContinu=20):
        
        seriesGraph = self[[features[0],features[1]]].dropna()
        
        # Calcul du coefficient de corrélation
        corr_rapport = eta_squared(seriesGraph[features[0]],seriesGraph[features[1]])
        
        if returnFig:
            return corr_rapport
        else:
            # liste des modalités de la variable qualitative
            modalites = seriesGraph[[features[0]]].sort_values(features[0], ascending=False)[features[0]].unique()

            # On splitte la variable quantitative en fonction de la variable qualitative
            groupes = []
            for m in modalites:
                groupes.append(seriesGraph[seriesGraph[features[0]]==m][features[1]])

            # Propriétés graphiques
            medianprops = {'color':"black"}
            meanprops = {'marker':'o', 'markeredgecolor':'black',
                        'markerfacecolor':'firebrick'}

            # Mise en graphique
            nbPlot = 1
            fig = plt.figure(figsize=(size * (18 * nbPlot/2), size * 8))

            plt.boxplot(groupes, labels=modalites, showfliers=boxShowfliers, medianprops=medianprops, 
                        vert=False, patch_artist=True, showmeans=True, meanprops=meanprops)
            plt.gca().set_xlabel(features[1])     
            plt.gca().set_ylabel(features[0])     

            plt.rcParams.update({'font.size':18})
            if pltTitle == "tbd":
                plt.suptitle("Statistique bivariée des champs " + features[0] + " et " + features[1])
            else:
                plt.suptitle(pltTitle)
            plt.rcParams.update({'font.size':10, 'font.style':'normal'})        
            plt.show()      
            
            print("Rapport de corrélation entre {} et {}: {}".format(features[0],features[1],str(corr_rapport)))
    
       
        
    def StatBivQualQual(self, features, size=1, statsNan=True, pltTitle="tbd", typeChart="tbd", typeFeature="tbd", threshold=2.5,  modalities=10.1, lower=False, truncateLabel=35, histDensity=True, histxLog=False, histyLog=False, histBins=100, pieColors="default", pieStartAngle=0.001, pieThresholdVoidLabel=2, pieThresholdOnlyPrctLabel=10, pieThresholdVoidModality=1,violinPlot=False, boxShowfliers=True, returnFig=False,
                        thresholdContinu=20):

        # c : contingence table
        c = self[[features[0],features[1]]].dropna().pivot_table(index=features[0],columns=features[1],aggfunc=len,margins=True,margins_name="Total")
        c = c.fillna(0)

        n = c.loc["Total","Total"]
        tx = c.loc[:,["Total"]]
        ty = c.loc[["Total"],:]
        
        # i : independant table
        i = tx.dot(ty) / n

        chi2Table = (c-i)**2/i
        chi2 = chi2Table.sum().sum()
        
        phi2 = chi2 / n
        r = c.shape[0] - 1
        c = c.shape[1] - 1
        tschuprow = math.sqrt(phi2 / math.sqrt((r-1)*(c-1)))
        cramer = math.sqrt(phi2/(min(r,c)-1))
        
        if returnFig:
            if np.isnan((tschuprow + cramer) / 2):
                return 0
            else:
                return (tschuprow + cramer) / 2
        else:
            nbPlot = 1
            fig = plt.figure(figsize=(size * (18 * nbPlot/2), size * 8))
            table = chi2Table/chi2   
            heatMap(table.iloc[:-1,:-1], colorGradient=[], cmap="YlGnBu")

            plt.rcParams.update({'font.size':18})
            if pltTitle == "tbd":
                plt.suptitle("Statistique bivariée (chi-2) des champs " + features[0] + " et " + features[1])
            else:
                plt.suptitle(pltTitle)
            plt.rcParams.update({'font.size':10, 'font.style':'normal'})        
            plt.show()


            print("chi-2 entre " + features[0] + " et "  + features[1] + " : {} ".format(chi2))
            print("phi-2 entre " + features[0] + " et "  + features[1] + " : {} ".format(phi2))
            print("coefficient T de Tschuprow entre " + features[0] + " et "  + features[1] + " : {} ".format(tschuprow))
            print("coefficient C de Cramer entre " + features[0] + " et "  + features[1] + " : {} ".format(cramer))
 

    def pca(self, nComp=3, reduction=True, labels=None, multivColorCat = None, multivGraph="Circle2D3D"):

    
        # liste des indivdus
        if labels == True:
            labels = self.index
        else:
            labels = None
            
        # Liste des variables
        features = self.columns.tolist()

        # Matrice des valeurs
        X = self.values
        
        # Variable catégorielle pour les couleurs
        if multivColorCat is not None:
            features.remove(multivColorCat)
            multivColor = X[:,-1]
            X = X[:,:-1]
        else:
            multivColor = None

        # Centrage et éventuellement réduction
        X = preprocessing.StandardScaler(with_std=reduction).fit_transform(X)
        # ou en deux temps:
        #X = preprocessing.StandardScaler().fit(X)
        #X = std_scale.transform(X)

        # Calcul des composantes principales
        pca = decomposition.PCA(n_components= nComp)
        pca.fit(X)
        
        # Projection des individus sur les composantes principales
        X_projected = pca.transform(X)
        
        # Eboulis des valeurs propres
        #pcaScreePlot(pca)
        
        # Cercle des corrélations des variables projetées
        if "circle" in multivGraph.lower():
            pcaCircles(pca, [(0,1),(0,2)], labels = np.array(features))
        
        # Visualisation 2D
        if "2d" in multivGraph.lower():
            pca2D(X_projected, pca, alpha = 0.2 ,labels=labels, multivColorCat=multivColor)
        
        # Visualisation 3D
        if "3d" in multivGraph.lower():
            pca3D(X_projected, pca, multivColorCat=multivColor)

 

    def Stat(self, col=["all_columns"], exclCol=[], returnTypeCol=False, bivInclCol=[], bivThresholdCorr=0, bivOnlyCorrTable= False, bivAllCorrTable=True, multivGraph="Circle2D3xD", multivHeatAnnot=True,
             multivDropNa=True, multivMeanNa=False, multivColorCat=None, multivProjNbIndiv=10000,
             typeCol="quantqual", typeStat="univ-biv-multiv", nCompACP=3, size=1, 
             statsNan=False, pltTitle="tbd", typeChart="tbd", typeFeature="tbd", threshold=2.5,  modalities=10.1, lower=False, truncateLabel=35, histDensity=True, 
             histxLog=False, histyLog=False, histBins=100, pieColors="default", pieStartAngle=0.001, pieThresholdVoidLabel=2, pieThresholdOnlyPrctLabel=10, pieThresholdVoidModality=1, 
             violinPlot=False, boxShowfliers=True, returnFig=False, thresholdContinu=20):
        
        ###############################################################
        # listes des statistiques à faire:
        #      - uniariée:un élément string
        #      - bivariée: une liste de deux éléments string
        ###############################################################

        # les champs colonne sont convertis en liste
        if type(col) == str:
            col = col.split(",")
            col = [elt.strip() for elt in col]
        if type(exclCol) == str:
            exclCol = exclCol.split(",")
            exclCol = [elt.strip() for elt in exclCol]
        if type(bivInclCol) == str:
            bivInclCol = bivInclCol.split(",")
            bivInclCol = [elt.strip() for elt in bivInclCol]   
        
        # liste des colonnes sur lesquelles les statistiques vont être faites
        if (col == ["all_columns"]): 
            col = self.columns.to_list()
            col = removeEltList(col, exclCol)   # pour exclure des colonnes
            
        # Les colonnes passées en paramètre doivent être des colonnes du dataframe
        for column in col:
            if column not in self.columns: raise ValueError(column + " n'est pas une colonne du dataframe")
        for column in exclCol:
            if column not in self.columns: raise ValueError(column + " n'est pas une colonne du dataframe")
        for column in bivInclCol:
            if column not in self.columns: raise ValueError(column + " n'est pas une colonne du dataframe")
                    

        # les colonnes de bivInclCol doivent être dans col (pour statistique bivariée)
        for column in bivInclCol:
            if column not in col:
                col.append(column)     
            
        # classification des variables (colonnes)
        features = pd.DataFrame(columns=["feature", "typeFeature"])
        for feature in col:
            features = features.append(self.ClassifyFeature(feature.strip(),typeFeature))            
        features = features.reset_index(drop=True)

        # liste des variables quantitatives
        if typeCol.lower().find("quant") >= 0:
            quantitativeFeatures = features.query("typeFeature == 'continue'").feature.values.tolist()
        else:
            quantitativeFeatures = []
            features = features.drop(features[features.typeFeature == "continue"].index)
            
        # liste des variables qualitatives
        if typeCol.lower().find("qual") >= 0:
            qualitativeFeatures = features.query("typeFeature != 'continue'").feature.values.tolist()
        else:
            qualitativeFeatures = []
            features = features.drop(features[features.typeFeature != "continue"].index)
        
        if returnTypeCol:
            return [qualitativeFeatures,quantitativeFeatures]
        
        # Statistiques univariées
        if typeStat.lower().find("univ") >= 0:
            for index, row in features.iterrows():
                myDf(self).StatUniv(row.feature, statsNan=statsNan,pltTitle=pltTitle,typeChart=typeChart,typeFeature=row.typeFeature,threshold=threshold,modalities=modalities,lower=lower,truncateLabel=truncateLabel,histDensity=histDensity,histBins=histBins,histxLog=histxLog,histyLog=histyLog,pieColors=pieColors,pieStartAngle=pieStartAngle,pieThresholdVoidLabel=pieThresholdVoidLabel,pieThresholdOnlyPrctLabel=pieThresholdOnlyPrctLabel,pieThresholdVoidModality=pieThresholdVoidModality,violinPlot=violinPlot, boxShowfliers=boxShowfliers,thresholdContinu=thresholdContinu)
        
        # Statistiques bivariées
        if typeStat.lower().find("biv") >= 0:        
            
            # bivInclCol ne doit pas mélanger variables qualitatives et quantitatives (sinon les tables de corrélations deviennent difficiles à gérer)
            if (len(commonEltList(bivInclCol,qualitativeFeatures)) > 0) & (len(commonEltList(bivInclCol,quantitativeFeatures)) > 0):
                warnings.warn("Le paramètre bivInclCol ne doit pas mélanger des variables quantitatives et des variables qualitatives.")
                bivInclCol = []                 
            
            corrFeatures = pd.DataFrame(columns=["feature1", "feature2", "typeFeature1", "typeFeature2", "correlation"])
            i=0  # pour incrémenter les index dans corrFeatures
            
            # Corrélations entre variables qualitatives et quantitatives
            for feature1 in qualitativeFeatures:
                for feature2 in quantitativeFeatures:
                    corrRapp = myDf(self).StatBivQualQuant([feature1,feature2],returnFig=True)
                    corrFeatures.loc[i] = [feature1, feature2, "qualitative", "quantitative", corrRapp]
                    i += 1     
            
            # Corrélations entre variables qualitatives
            j=0
            for feature1 in qualitativeFeatures:
                for feature2 in qualitativeFeatures[j:len(qualitativeFeatures)]:
                    if feature1 == feature2:
                        tschuprow = 1
                    else:
                        tschuprow = myDf(self).StatBivQualQual([feature1,feature2],returnFig=True)
                    corrFeatures.loc[i] = [min(feature1, feature2), max(feature1, feature2), "qualitative", "qualitative", tschuprow]
                    i += 1
                j += 1

            # Corrélations entre variables quantitatives
            corrMatrix = self[quantitativeFeatures].corr()
            for index, row in corrMatrix.iterrows():
                for col in corrMatrix.columns:
                    corrFeatures.loc[i] = [min(index, col), max(col, index), "quantitative", "quantitative", row[col]]
                    i += 1
        
            # Suppression des corrélations pour lesquelles aucune variable n'est listée dans bivInclCol
            if bivInclCol != []:
                listIndexToDrop = []
                for index, row in corrFeatures.iterrows():
                    if row.feature1 not in bivInclCol and row.feature2 not in bivInclCol:
                        listIndexToDrop.append(index)
                corrFeatures = corrFeatures.drop(listIndexToDrop)                   
        
            # Suppression des doublons, ordonnancement décroissant en valeur absolue, remplacement des NaN par 1
            corrFeatures = corrFeatures.drop_duplicates()
            corrFeatures["correlationAbs"] = corrFeatures.apply(lambda x: abs(x.correlation), axis = 1)
            corrFeatures = corrFeatures.sort_values("correlationAbs", ascending=False)
            del corrFeatures["correlationAbs"]
            corrFeatures = corrFeatures.fillna(1)
            corrFeatures = corrFeatures.reset_index(drop=True)         
            
            # MATRICE DE TOUTES LES CORRELATIONS
            if bivAllCorrTable:
                if bivOnlyCorrTable | (((len(qualitativeFeatures) * len(quantitativeFeatures)) != 0) & (typeCol != "quant-qual") & (typeCol != "qual-quant")):
                    # Création de la matrice
                    if len(bivInclCol) > 0:
                        inds = bivInclCol
                        cols = bivInclCol + removeEltList(qualitativeFeatures+quantitativeFeatures,bivInclCol)
                    else:
                        inds = qualitativeFeatures+quantitativeFeatures
                        cols = qualitativeFeatures+quantitativeFeatures
                    corrMatrix = pd.DataFrame(index=inds,columns=cols)
                    for index,row in corrFeatures.iterrows():
                        if row.feature1 in corrMatrix.index: corrMatrix.at[row.feature1,row.feature2] = row.correlation
                        if row.feature2 in corrMatrix.index: corrMatrix.at[row.feature2,row.feature1] = row.correlation
                    corrMatrix = corrMatrix.fillna(1)

                    # Affichage d'une heat map de la matrice
                    if ((corrMatrix.shape[0] > 2) | (corrMatrix.shape[1] > 2)) & (corrMatrix.shape[0] * corrMatrix.shape[1] != 0):
                        f, ax = plt.subplots(figsize=(10, 8))
                        plt.rcParams.update({'font.size':10, 'font.style':'normal'})  
                        heatMap(corrMatrix, square=True, annot=multivHeatAnnot)
                        plt.rcParams.update({'font.size':18})
                        if pltTitle == "tbd":
                            plt.suptitle("Matrice des corrélations (linéaire,  Tschuprow, rapport de corrélation)")
                        else:
                            plt.suptitle(pltTitle)
                        plt.rcParams.update({'font.size':10, 'font.style':'normal'})                
                        plt.show()
            
            if not bivOnlyCorrTable:
            
                # STATISTIQUES ENTRE VARIABLES QUALITATIVES

                # Création de la matrice de Tschuprow entre les variables qualitatives
                if len(qualitativeFeatures) >= 2:

                    # Création des colonnes la matrice de Tschuprow
                    if len(bivInclCol) > 0:
                        if len(commonEltList(qualitativeFeatures,bivInclCol)) > 0:
                            inds = commonEltList(qualitativeFeatures,bivInclCol)
                            cols = commonEltList(qualitativeFeatures,bivInclCol) + removeEltList(qualitativeFeatures,bivInclCol)
                        else:
                            inds = []
                            cols = []
                    else:
                        inds = qualitativeFeatures
                        cols = qualitativeFeatures

                    corrTmp = corrFeatures.query("(typeFeature1 == 'qualitative') & (typeFeature2 == 'qualitative')")

                    # Création de la matrice de Tschuprow
                    corrMatrix = pd.DataFrame(index=inds, columns=cols)
                    for index, row in corrTmp.iterrows():
                        if row.feature1 in corrMatrix.index: corrMatrix.at[row.feature1,row.feature2] = row.correlation
                        if row.feature2 in corrMatrix.index: corrMatrix.at[row.feature2,row.feature1] = row.correlation 
                    corrMatrix = corrMatrix.fillna(1)               

                    if ((corrMatrix.shape[0] > 2) | (corrMatrix.shape[1] > 2)) & (corrMatrix.shape[0] * corrMatrix.shape[1] != 0)  & (typeCol != "quant-qual") & (typeCol != "qual-quant"):
                        # Affichage de la matrice
                        f, ax = plt.subplots(figsize=(10, 8))
                        plt.rcParams.update({'font.size':10, 'font.style':'normal'})  
                        heatMap(corrMatrix, square=True, annot=multivHeatAnnot)
                        plt.rcParams.update({'font.size':18})
                        if pltTitle == "tbd":
                            plt.suptitle("Matrice de Tschuprow (variables qualitatives)")
                        else:
                            plt.suptitle(pltTitle)
                        plt.rcParams.update({'font.size':10, 'font.style':'normal'})                
                        plt.show() 

                    # Affichage des statistiques bivariées
                    if  (typeCol != "quant-qual") & (typeCol != "qual-quant"):
                        for index, row in corrTmp.iterrows():
                            if (row["feature1"] != row["feature2"]) & (row["correlation"] >= bivThresholdCorr):
                                myDf(self).StatBivQualQual([row["feature1"], row["feature2"]], statsNan=statsNan,pltTitle=pltTitle,typeChart=typeChart,typeFeature=typeFeature,threshold=threshold,modalities=modalities,lower=lower,truncateLabel=truncateLabel,histDensity=histDensity,histBins=histBins,pieColors=pieColors,pieStartAngle=pieStartAngle,pieThresholdVoidLabel=pieThresholdVoidLabel,pieThresholdOnlyPrctLabel=pieThresholdOnlyPrctLabel,pieThresholdVoidModality=pieThresholdVoidModality,violinPlot=violinPlot, boxShowfliers=boxShowfliers,thresholdContinu=thresholdContinu)


                
                # STATISTIQUES ENTRE VARIABLES QUALITATIVES ET QUANTITATIVES

                # Création de la matrice des rapports de corrélation entre les variables quantitatives et qualitatives
                if (len(quantitativeFeatures) + len(qualitativeFeatures) >= 2) & (len(qualitativeFeatures) > 0) & (len(quantitativeFeatures) > 0):

                    # Création des colonnes la matrice
                    if len(bivInclCol) > 0:
                        inds = bivInclCol
                        if len(commonEltList(quantitativeFeatures,bivInclCol)) > 0:
                            cols = bivInclCol + qualitativeFeatures
                        else:
                            cols = bivInclCol + quantitativeFeatures
                    else:
                        inds = qualitativeFeatures
                        cols = quantitativeFeatures

                    corrTmp = corrFeatures.query("typeFeature1 != typeFeature2")
                    corrTmp = corrTmp.reset_index(drop=True)
                    for index, row in corrFeatures.iterrows():
                        if (row.feature1 in bivInclCol) & (row.feature2 in bivInclCol):
                            corrTmp.loc[len(corrTmp)] = [row.feature1, row.feature2, row.typeFeature1, row.typeFeature1, row.correlation]

                    # Création de la matrice des rapports de corrélation
                    corrMatrix = pd.DataFrame(index=inds, columns=cols)
                    for index, row in corrTmp.iterrows():
                        if row.feature1 in corrMatrix.index: corrMatrix.at[row.feature1,row.feature2] = row.correlation
                        if row.feature2 in corrMatrix.index: corrMatrix.at[row.feature2,row.feature1] = row.correlation 

                    corrMatrix = corrMatrix.fillna(1)               

                    if ((corrMatrix.shape[0] >= 2) | (corrMatrix.shape[1] >= 2)) & (corrMatrix.shape[0] * corrMatrix.shape[1] != 0):
                        # Affichage de la matrice
                        f, ax = plt.subplots(figsize=(10, 8))
                        heatMap(corrMatrix, square=True, annot=multivHeatAnnot)
                        plt.rcParams.update({'font.size':18})
                        if pltTitle == "tbd":
                            plt.suptitle("Matrice des rapports de corrélation entre variables qualitatives et quantitatives")
                        else:
                            plt.suptitle(pltTitle)
                        plt.rcParams.update({'font.size':10, 'font.style':'normal'})                
                        plt.show() 

                    # Affichage des statistiques bivariées
                    for index, row in corrTmp.iterrows():
                        if (row["feature1"] != row["feature2"]) & (row["typeFeature1"] != row["typeFeature2"]) & (row["correlation"] >= bivThresholdCorr):
                            myDf(self).StatBivQualQuant([row["feature1"], row["feature2"]], statsNan=statsNan,pltTitle=pltTitle,typeChart=typeChart,typeFeature=typeFeature,threshold=threshold,modalities=modalities,
                                                        lower=lower,truncateLabel=truncateLabel,histDensity=histDensity,histBins=histBins,pieColors=pieColors,pieStartAngle=pieStartAngle,
                                                        pieThresholdVoidLabel=pieThresholdVoidLabel,pieThresholdOnlyPrctLabel=pieThresholdOnlyPrctLabel,pieThresholdVoidModality=pieThresholdVoidModality,
                                                        violinPlot=violinPlot, boxShowfliers=boxShowfliers,thresholdContinu=thresholdContinu)

                            
                                            # STATISTIQUES ENTRE VARIABLES QUANTITATIVES

                # Création de la matrice des coefficients de corrélation entre les variables quantitatives
                if len(quantitativeFeatures) >= 2:

                    # Création des colonnes la matrice de Tschuprow
                    if len(bivInclCol) > 0:
                        if len(commonEltList(quantitativeFeatures,bivInclCol)) > 0:
                            inds = commonEltList(quantitativeFeatures,bivInclCol)
                            cols = commonEltList(quantitativeFeatures,bivInclCol) + removeEltList(quantitativeFeatures,bivInclCol)
                        else:
                            inds = []
                            cols = []
                    else:
                        inds = quantitativeFeatures
                        cols = quantitativeFeatures

                    corrTmp = corrFeatures.query("(typeFeature1 == 'quantitative') & (typeFeature2 == 'quantitative')")

                    # Création de la matrice des coefficients de corrélation
                    corrMatrix = pd.DataFrame(index=inds, columns=cols)
                    for index, row in corrTmp.iterrows():
                        if row.feature1 in corrMatrix.index: corrMatrix.at[row.feature1,row.feature2] = row.correlation
                        if row.feature2 in corrMatrix.index: corrMatrix.at[row.feature2,row.feature1] = row.correlation 
                    corrMatrix = corrMatrix.fillna(1)           

                    if ((corrMatrix.shape[0] > 2) | (corrMatrix.shape[1] > 2)) & (corrMatrix.shape[0] * corrMatrix.shape[1] != 0)  & (typeCol != "quant-qual") & (typeCol != "qual-quant"):
                        # Affichage de la matrice
                        f, ax = plt.subplots(figsize=(10, 8))
                        heatMap(corrMatrix, square=True, annot=multivHeatAnnot)
                        plt.rcParams.update({'font.size':18})
                        if pltTitle == "tbd":
                            plt.suptitle("Matrice des coefficients de corrélation (variables quantitatives)")
                        else:
                            plt.suptitle(pltTitle)
                        plt.rcParams.update({'font.size':10, 'font.style':'normal'})                
                        plt.show()

                    # Affichage des statistiques bivariées
                    if  (typeCol != "quant-qual") & (typeCol != "qual-quant"): 
                        for index, row in corrTmp.iterrows():             
                            if (row["feature1"] != row["feature2"]) & (abs(row["correlation"]) >= bivThresholdCorr):
                                myDf(self).StatBivQuantQuant([row["feature1"], row["feature2"]], statsNan=statsNan,pltTitle=pltTitle,typeChart=typeChart,typeFeature=typeFeature,threshold=threshold,modalities=modalities,lower=lower,truncateLabel=truncateLabel,histDensity=histDensity,histBins=histBins,pieColors=pieColors,pieStartAngle=pieStartAngle,pieThresholdVoidLabel=pieThresholdVoidLabel,pieThresholdOnlyPrctLabel=pieThresholdOnlyPrctLabel,pieThresholdVoidModality=pieThresholdVoidModality,violinPlot=violinPlot, boxShowfliers=boxShowfliers,thresholdContinu=thresholdContinu)

                                
                                
        # Statistique mutlivariée (PCA sur 3 composantes principales)
        if typeStat.lower().find("multiv") >= 0: 

            if len(quantitativeFeatures) >= 3:
                
                data_pca = self.copy()
                columnsPca = quantitativeFeatures
                
                if multivColorCat is not None:
                    if multivColorCat in self.columns:
                        data_pca = data_pca.dropna(subset=[multivColorCat])
                        columnsPca.append(multivColorCat)
                    else:
                        warnings.warn(multivColorCat + " n'est pas une colonne du dataframe")
                        multivColorCat = None

                data_pca = data_pca[columnsPca]
                                                       
                if multivDropNa:
                    data_pca = data_pca.dropna(subset=quantitativeFeatures)
                elif multivMeanNa:
                    data_pca[quantitativeFeatures] = self[quantitativeFeatures].fillna(data_pca.mean()) 
                    
                # On prend max 10000 lignes
                data_pca = data_pca.reset_index(drop=True)
                multivProjNbIndiv = min(multivProjNbIndiv,data_pca.shape[0])
                data_pca = data_pca.iloc[random.sample(data_pca.index.tolist(), min(multivProjNbIndiv,len(data_pca)))]

                myDf(data_pca).pca(multivColorCat=multivColorCat, multivGraph=multivGraph)
    
            
            
                        
                


      
