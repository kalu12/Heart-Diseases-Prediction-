
from ucitavanje import *



N=1000
Ntrening=int(0.6*1000)


X1 = data.loc[data['HeartDisease']==1,['GenHealth','AgeCategory','DiffWalking','PhysicalHealth','Diabetic','Stroke']]


X2 = data.loc[data['HeartDisease']==0,['GenHealth','AgeCategory','DiffWalking','PhysicalHealth','Diabetic','Stroke']]
plt.figure()
plt.hist(X1)
plt.title('Klasa 1')
plt.show()
plt.figure()
plt.hist(X2)
plt.title('Klasa 2')
plt.show()

Ntrening1 = int(27373*0.6)
X1_trening = X1.head(Ntrening1)
X1_test = X1.head(-Ntrening1)

#print(max(principal_df['LDA1'].astype(float)))

Ntrening2 = int(292422*0.6)
X2_trening = X2.head(Ntrening2)
X2_test = X2.head(-Ntrening2)


M1p = np.mean(X1_trening, axis=0)
S1p = np.cov(X1_trening.T)

M2p = np.mean(X2_trening, axis=0)
S2p = np.cov(X2_trening.T)



print(M1p)
print(S1p)

print(M2p)
print(S2p)



def izracunaj_fgv(x, m, s):
    det = np.linalg.det(s)
    #print(s)
    inv = np.linalg.inv(s)
    x_mu = x - m
    #print(inv)
    fgv_const = 1/np.sqrt(2*np.pi*det)
    fgv_rest = np.exp(-0.5*x_mu.T@inv@x_mu)
    #print(inv)
    return fgv_const*fgv_rest




p1=np.shape(X1)[0]/(np.shape(X1)[0]+np.shape(X2)[0])
p2=np.shape(X2)[0]/(np.shape(X1)[0]+np.shape(X2)[0])
T=np.log(p1/p2)
odluka=np.zeros((800,1))
conf_mat=np.zeros((2,2))
for i in range(N-Ntrening):
    X = X1_test.iloc[i].to_numpy()
    f1=izracunaj_fgv(X, M1p, S1p)
    f2=izracunaj_fgv(X, M2p, S2p)
    h1=-np.log(f1)+np.log(f2)
    if h1<T:
        odluka[i]=0
    else:
        odluka[i]=1
for i in range(N-Ntrening):
    x2 = X2_test.iloc[i].to_numpy()
    f1=izracunaj_fgv(x2,M1p,S1p)
    f2=izracunaj_fgv(x2,M2p,S2p)
    h2=-np.log(f1)+np.log(f2)
    if h2<T:
        odluka[N-Ntrening+i]=0
    else:
        odluka[N-Ntrening+i]=1
Xtest=np.append(X1_test,X2_test,axis=0)
Ytest=np.append(np.zeros((400,1)),np.ones((400,1)))
from sklearn.metrics import confusion_matrix
conf_mat=confusion_matrix(Ytest,odluka)
plt.figure()
sns.heatmap(conf_mat,annot=True,fmt='g',cbar=False)
plt.show()
