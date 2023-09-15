# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 23:45:42 2022

@author: kaluh
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 18:21:39 2022

@author: kaluh
"""

from ucitavanje import *



N=10000
Ntrening=int(0.6*10000)

X1 = data.loc[data['HeartDisease']==1,['GenHealth']]
X2 = data.loc[data['HeartDisease']==0,['GenHealth']]
X11 = data.loc[data['HeartDisease']==1,['Diabetic']]
X22 = data.loc[data['HeartDisease']==0,['Diabetic']]


X2i = X2[10000:15000]
X1i=X1[10000:15000]
X11i =X11[10000:15000]
X22i=X22[10000:15000]
X1i = X1i.to_numpy() 
X11i = X11i.to_numpy() 
X2i = X2i.to_numpy() 
X22i = X22i.to_numpy() 


X2 = X2[:10000]
X1=X1[:10000]
X11 =X11[:10000]
X22=X22[:10000]





n=np.zeros(10000)
j=np.ones(10000)
j1=j.reshape(10000,1)
n1=n.reshape(10000,1)
n=np.zeros(10000)
x1=X1['GenHealth']
x2=X2['GenHealth']
X22=X22['Diabetic']
X11=X11['Diabetic']


x11 = x1.to_numpy() 
X11 = X11.to_numpy()
X22 =X22.to_numpy()  
x11=x11.reshape(10000,1)
X11=X11.reshape(10000,1)
x22 = x2.to_numpy() 
X22=X22.reshape(10000,1)

x22=x22.reshape(10000,1)




X111 = data.loc[data['HeartDisease']==1,['Stroke']]
X222 = data.loc[data['HeartDisease']==0,['Stroke']]
X1111 = data.loc[data['HeartDisease']==1,['AgeCategory']]
X2222 = data.loc[data['HeartDisease']==0,['AgeCategory']]


X222i = X222[10000:15000]
X111i=X111[10000:15000]
X1111i = X1111[10000:15000]
X2222i=X2222[10000:15000]
X111i = X111i.to_numpy() 
X1111i = X1111i.to_numpy() 
X2222i = X2222i.to_numpy() 
X222i = X222i.to_numpy() 



X222 = X222[:10000]
X111=X111[:10000]
X1111 = X1111[:10000]
X2222=X2222[:10000]


n=np.zeros(10000)
j=np.ones(10000)
j1=j.reshape(10000,1)
n1=n.reshape(10000,1)
n=np.zeros(10000)
x111=X111['Stroke']
x222=X222['Stroke']
X1111=X1111['AgeCategory']
X2222=X2222['AgeCategory']

x111 = x111.to_numpy() 
X1111 = X1111.to_numpy()
X2222 =X2222.to_numpy()  

x111=x111.reshape(10000,1)
X1111=X1111.reshape(10000,1)
x222 = x222.to_numpy() 
X2222=X2222.reshape(10000,1)

x222=x222.reshape(10000,1)



arr=np.append(x11,X11,axis=1)
arr1=np.append(x22,X22,axis=1)
arr2=np.append(x111,X1111,axis=1)
arr3=np.append(x222,X2222,axis=1)

arr4=np.append(arr,arr2,axis=1).T
arr5=np.append(arr1,arr3,axis=1).T

Z1=np.append(-arr4,-np.ones((1,N)),axis=0)
Z2=np.append(arr5,np.ones((1,N)),axis=0)
U=np.append(Z1,Z2,axis=1)
Gama=np.ones((2*N,1))
W=np.linalg.inv(U@U.T)@U@Gama
V=W[:4]
V0=W[4]

conf_mat=np.zeros((2,2))
prom=[]
for i in range(10000):
    #prom[i]=V[0]*x11[i]+V[1]*X11[i]+V[2]*x111[i]+V[3]*X1111[i]+V0
    if((V[0]*x11[i]+V[1]*X11[i]+V[2]*x111[i]+V[3]*X1111[i]+V0)<0):
        conf_mat[0,0]+=1
    else:
        conf_mat[1,0]+=1

    if((V[0]*x22[i]+V[1]*X22[i]+V[2]*x222[i]+V[3]*X2222[i]+V0)>0):
        conf_mat[1,1]+=1
    else:
        conf_mat[0,1]+=1

plt.figure()
sns.heatmap(conf_mat,annot=True,fmt='g',cbar=False)
plt.show()


print(X1i[4999])

conf_mati=np.zeros((2,2))
prom=[]
for i in range(len(X1i)):
    #prom[i]=V[0]*x11[i]+V[1]*X11[i]+V[2]*x111[i]+V[3]*X1111[i]+V0
    if((V[0]*X1i[i]+V[1]*X11i[i]+V[2]*X111i[i]+V[3]*X1111i[i]+V0)<0):
        conf_mati[0,0]+=1
    else:
        conf_mati[1,0]+=1

    if((V[0]*X2i[i]+V[1]*X22i[i]+V[2]*X222i[i]+V[3]*X2222i[i]+V0)>0):
        conf_mati[1,1]+=1
    else:
        conf_mati[0,1]+=1

plt.figure()
sns.heatmap(conf_mati,annot=True,fmt='g',cbar=False)
plt.show()


        
