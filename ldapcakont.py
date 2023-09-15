from ucitavanje import *


#normalizacija atributa
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_norm = X -np.mean(X , axis = 0)
X_norm /= np.max(X_norm, axis = 0)


#racunanje kovariacione matrice
#racunanje sopstvenih vrednosti i sopstvenih vektora
Sx = np.cov(X_norm.T)
eigval, eigvec = np.linalg.eig(Sx)
idx = np.argsort(eigval)[::-1]
eigval = eigval[idx]
eigvec = eigvec[:, idx]
plt.figure()
plt.bar(np.arange(len(eigval))+1, eigval)
plt.show()


#primena transformacije
no_comp=2
A=eigvec[:,:no_comp]
Y=A.T@X_norm.T
Y=Y.T
principal_df=pd.concat([Y,y],axis=1)
principal_df.columns=['PC1','PC2','Outcome']
principal_df.head()

#prikaz
plt.figure()
sns.scatterplot(data=principal_df,x='PC1',y='PC2',hue='Outcome')
plt.show()

from sklearn.decomposition import PCA
pca=PCA(n_components=no_comp)
principalComponents=pca.fit_transform(X_norm)
principalComponents=pd.DataFrame(data=principalComponents)
principal_df2=pd.concat([principalComponents,y],axis=1)
principal_df2.columns=['PC1','PC2','Outcome']
principal_df2.head()


plt.figure()
sns.scatterplot(data=principal_df2,x='PC1',y='PC2',hue='Outcome')
plt.show()


#ROC
pca_var=eigval/np.sum(eigval)
cum_sum_eigenvalues=np.cumsum(pca_var)
plt.bar(range(len(pca_var)),pca_var)
plt.plot(range(len(cum_sum_eigenvalues)),cum_sum_eigenvalues,'r*')
plt.show()


X0 = X_norm.loc[y==0, :]
p0 = X0.shape[0]/X_norm.shape[0]
M0 = X0.mean().values.reshape(X0.shape[1],1)
S0 = X0.cov()
X1 = X_norm.loc[y==1, :]
p1 = X1.shape[0]/X_norm.shape[0]
M1 = X1.mean().values.reshape(X1.shape[1],1)
S1 = X1.cov()
  


M = p1*M1 + p0*M0
Sw = p1*S1 + p0*S0
Sb = p1*(M1-M)@(M1-M).T + p0*(M0-M)@(M0-M).T 
Sm = Sb + Sw
T = np.linalg.inv(Sw)@Sb
eigval, eigvec = np.linalg.eig(T)
plt.figure()
plt.bar(np.arange(len(eigval))+1, eigval)
plt.show()



no_comp = 1
A = eigvec[:, :no_comp]
Y = A.T @ X_norm.T
Y = Y.T
nula_niz=np.zeros(319795)
nula_niz = pd.Series(nula_niz)
principal_df1 = pd.concat([Y ,nula_niz] , axis = 1)
principal_df = pd.concat([principal_df1 ,y] , axis = 1)
principal_df.columns = ['LDA' ,'nula_niz','Outcome']
principal_df.head()
dff = principal_df
#dff['Outcome']=pd.Categorical(dff.n)
plt.figure()
sns.scatterplot(data =dff, x = 'LDA' ,y='nula_niz' , hue='Outcome')
plt.show()


#ROC
pca_var=eigval/np.sum(eigval)
cum_sum_eigenvalues=np.cumsum(pca_var)
plt.bar(range(len(pca_var)),pca_var)
plt.plot(range(len(cum_sum_eigenvalues)),cum_sum_eigenvalues,'r*')
plt.show()



