import numpy as np
import kmeans
import common
import naive_em
import em
import ems
X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt('netflix_complete.txt')
L=[]
for i in range(5):
    mixt,post=common.init(X,12,i)
    mixt,post,l=kmeans.run(X,mixt,post)
    L.append(l)
    print(str(i)+' : ',l)
#i=L.index(max(L))
#mixt,post=common.init(X,12,1)
#mixt,post,l=em.run(X,mixt,post)
#X_pred=em.fill_matrix(X,mixt)
#error=common.rmse(X_gold,X_pred)
