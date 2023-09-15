from ucitavanje import*

from keras.models import Sequential 
from keras.layers import Dense 
from keras.utils import to_categorical
from keras import regularizers
 




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
X11111 = data.loc[data['HeartDisease']==1,['Diabetic']]
X22222 = data.loc[data['HeartDisease']==0,['Diabetic']]
plt.plot(x11,X11,'.')
plt.plot(X22,x22,'.')



X222 = X222[:1000]
X111=X111[:1000]
X1111 = X1111[:1000]
X2222=X2222[:1000]
X11111 = X11111[:1000]
X22222=X22222[:1000]

n=np.zeros(1000)
j=np.ones(1000)
j1=j.reshape(1000,1)
n1=n.reshape(1000,1)
n=np.zeros(1000)
x111=X111['Stroke']
x222=X222['Stroke']
X1111=X1111['PhysicalHealth']
X2222=X2222['PhysicalHealth']
X11111=X11111['Diabetic']
X22222=X22222['Diabetic']
x111 = x111.to_numpy() 
X1111 = X1111.to_numpy()
X2222 =X2222.to_numpy()  
X11111 = X11111.to_numpy()
X22222 =X22222.to_numpy()  
x111=x111.reshape(1000,1)
X1111=X1111.reshape(1000,1)
X11111=X11111.reshape(1000,1)
x222 = x222.to_numpy() 
X2222=X2222.reshape(1000,1)
X22222=X22222.reshape(1000,1)

x222=x222.reshape(1000,1)


K11=np.append(x11,X11,axis=1)
K22=np.append(x22,X22,axis=1)
K111=np.append(x111,X1111,axis=1)
K222=np.append(x222,X2222,axis=1)
#K1111=np.append(K11,K111,axis=1)
#K2222=np.append(K22,K222,axis=1)
K1=np.append(K11,K111,axis=1)
K2=np.append(K22,K222,axis=1)


X=np.append(K2,K1,axis=0)
Y=np.append(np.zeros((N,1)),np.ones((N,1)),axis=0)

from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,train_size=0.8,random_state=42)

model=Sequential()
model.add(Dense(100,input_dim=4,activation='relu'))
model.add(Dense(1000,activation='relu'))
model.add(Dense(1000,activation='relu'))
model.add(Dense(1000,activation='relu'))



model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history=model.fit(Xtrain,Ytrain,epochs=100,validation_data=(Xtest,Ytest),verbose=0)
_,train_acc=model.evaluate(Xtrain,Ytrain,verbose=0)
print('Tačnost na trening skupu iznosi: '+str(train_acc*100)+'%.')
_,test_acc=model.evaluate(Xtest,Ytest,verbose=0)
print('Tačnost na test skupu iznosi: '+str(test_acc*100)+'%.')


plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["Trening skup","Validacioni skup"])
plt.title('Kriterijumska funkcija')



'''from sklearn.metrics import confusion_matrix
Ypred=model.predict(Xtest)
conf_mat=confusion_matrix(Ytest,Ypred)
import seaborn as sns 
sns.heatmap(conf_mat,annot=True,fmt='g',cbar=False)
plt.show()'''