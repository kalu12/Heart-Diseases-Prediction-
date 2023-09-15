from ucitavanje import *



N=1000
Ntrening=int(0.6*1000)

X1 = data.loc[data['HeartDisease']==1,['AgeCategory']]
X2 = data.loc[data['HeartDisease']==0,['AgeCategory']]
X11 = data.loc[data['HeartDisease']==1,['GenHealth']]
X22 = data.loc[data['HeartDisease']==0,['GenHealth']]





X2 = X2[:1000]
X1=X1[:1000]
X11 = X11[:1000]
X22=X22[:1000]


n=np.zeros(1000)
j=np.ones(1000)
j1=j.reshape(1000,1)
n1=n.reshape(1000,1)
n=np.zeros(1000)
x1=X1['AgeCategory']
x2=X2['AgeCategory']
X11=X11['GenHealth']
X22=X22['GenHealth']
x11 = x1.to_numpy() 
X11 = X11.to_numpy()
X22 =X22.to_numpy()  
x11=x11.reshape(1000,1)
X11=X11.reshape(1000,1)
x22 = x2.to_numpy() 
X22=X22.reshape(1000,1)

x22=x22.reshape(1000,1)

X111 = data.loc[data['HeartDisease']==1,['Stroke']]
X222 = data.loc[data['HeartDisease']==0,['Stroke']]
X1111 = data.loc[data['HeartDisease']==1,['PhysicalHealth']]
X2222 = data.loc[data['HeartDisease']==0,['PhysicalHealth']]

plt.plot(x11,X11,'.')
plt.plot(X22,x22,'.')



X222 = X222[:1000]
X111=X111[:1000]
X1111 = X1111[:1000]
X2222=X2222[:1000]


n=np.zeros(1000)
j=np.ones(1000)
j1=j.reshape(1000,1)
n1=n.reshape(1000,1)
n=np.zeros(1000)
x111=X111['Stroke']
x222=X222['Stroke']
X1111=X1111['PhysicalHealth']
X2222=X2222['PhysicalHealth']
x111 = x111.to_numpy() 
X1111 = X1111.to_numpy()
X2222 =X2222.to_numpy()  
x111=x111.reshape(1000,1)
X1111=X1111.reshape(1000,1)
x222 = x222.to_numpy() 
X2222=X2222.reshape(1000,1)

x222=x222.reshape(1000,1)




arr=np.append(x11,X11,axis=1)
arr1=np.append(x22,X22,axis=1)
arr2=np.append(x111,X1111,axis=1)
arr3=np.append(x222,X2222,axis=1)
arr4=np.append(arr,arr2,axis=1).T
arr5=np.append(arr1,arr3,axis=1).T

#USOEO SAM DA IZBEGNEM SINGULAR MATRIX ERROR MENJENJEM OBELEZJA
#Z1=np.concatenate((-x11**2,-x11*X11,-x11*x111,-x11*X1111,-X11**2,-X11*x111,-X11*X1111,-x111**2,-x111*X1111,-X1111**2,-x11,-X11,-x111,-X1111,-np.ones((N,1))),axis=1)
#Z2=np.concatenate((x22**2,x22*X22,x22*x222,x22*X2222,X22**2,X22*x222,X22*X2222,x222**2,x222*X2222,X2222**2,x22,X22,x222,X2222,np.ones((N,1))),axis=1)
Z1=np.concatenate((-x11**2,-x11*X11,-x11*x111,-X11**2,-X11*x111,-x111**2,-x11,-X11,-x111,-np.ones((N,1))),axis=1)
Z2=np.concatenate((x22**2,x22*X22,x22*x222,X22**2,X22*x222,x222**2,x22,X22,x222,np.ones((N,1))),axis=1)
#Z1=np.concatenate((-x11**2,-x11*X11,-X11**2,-x11,-X11,-np.ones((N,1))),axis=1)
#Z2=np.concatenate((x22**2,x22*X22,X22**2,x22,X22,np.ones((N,1))),axis=1)
U=np.concatenate((Z1,Z2),axis=0).T
Gama=np.ones((2*N,1))
W=np.linalg.inv(U@U.T)@U@Gama
q11=W[0]
q12=q21=W[1]/2
q13=q31=W[2]/2
q22=W[3]
q23=q32=W[4]/2
q33=W[5]
v1=W[6]
v2=W[7]
v3=W[8]
v0=W[9]

conf_mat=np.zeros((2,2))
prom=[]
for i in range(1000):
    #prom[i]=V[0]*x11[i]+V[1]*X11[i]+V[2]*x111[i]+V[3]*X1111[i]+V0
    if((x11[i]**2*q11+2*x11[i]*X11[i]*q12+2*x11[i]*x111[i]*q13+X11[i]**2*q22+2*X11[i]*x111[i]*q23+x111[i]**2*q33+x11[i]*v1+X11[i]*v2+x111[i]*v3+v0)>0):
        conf_mat[0,0]+=1
    else:
        conf_mat[1,0]+=1
    if((x22[i]**2*q11+2*x22[i]*X22[i]*q12+2*x22[i]*x222[i]*q13+X22[i]**2*q22+2*X22[i]*x222[i]*q23+x222[i]**2*q33+x22[i]*v1+X22[i]*v2+x222[i]*v3+v0)>0):
        conf_mat[1,1]+=1
    else:
        conf_mat[0,1]+=1
    
