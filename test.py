import numpy as np
import kmeans
import naive_em
import common

X = np.loadtxt('toy_data.txt')
l1=[]
for k in range(1,5):
    l2=[]
    l3=[]
    for i in range(5):
        mixture,post=common.init(X,k,i)
        cost=kmeans.run(X,mixture,post)[2]
        l2.append(cost)
        l3.append(L)
    l1.append((max(l2),max(l3)))
x,y=max(l1)
        
        
        
