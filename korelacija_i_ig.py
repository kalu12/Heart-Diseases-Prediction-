from ucitavanje import *




#pirsnov koeficijent korelacije
correlation = data.corr().round(2)
plt.figure(figsize = (14,7))
sns.heatmap(correlation, annot = True, cmap = 'YlOrBr')
plt.title("pearson")
plt.show()

sns.set_style('white')
sns.set_palette('YlOrBr')
plt.figure(figsize = (13,6))
plt.title('Distribution of correlation of features')
abs(correlation['HeartDisease']).sort_values()[:-1].plot.barh()
plt.show()




#spirmanov koeficijent korealicje
spearman_R=data.corr(method='spearman')
plt.figure(figsize = (14,7))
sns.heatmap(spearman_R,annot=True,cmap = 'YlOrBr')
plt.title("spearman")
plt.show()


#obelezja sa najvecim koef korelacije poredjani (pirson)
sns.set_style('white')
sns.set_palette('YlOrBr')
plt.figure(figsize = (13,6))
plt.title('Distribution of correlation of features')
abs(correlation['HeartDisease']).sort_values()[:-1].plot.barh()
plt.show()


#racunanje IG
def calculateInfoD(col):
    
    un = np.unique(col)
    infoD = 0
    
    for u in un:
        p = sum(col == u)/len(col)
        infoD -= p*np.log2(p)
    return infoD


klasa = data.iloc[:,-1]
infoD = calculateInfoD(klasa)
print('Info(D) = ' + str(infoD))








new_data = data.copy(deep = True)
def limit_feature(col, noSteps = 30):
    step = (max(col) - min(col))/noSteps
    new_col = np.floor(col/step)*step
    return new_col
for ob in range(1, data.shape[1]-1):
    temp = data.iloc[:, ob]
    new_data.iloc[:, ob] = limit_feature(temp)


IG = np.zeros((new_data.shape[1]-1, 2))
for ob in range(new_data.shape[1]-1):
    f = np.unique(new_data.iloc[:, ob])
 
    infoDA = 0
    for i in f:
        temp = klasa[new_data.iloc[:, ob] == i]
        infoDi = calculateInfoD(temp)
        Di = sum(new_data.iloc[:, ob] == i)
        D = len(new_data.iloc[:, ob])
        infoDA += Di*infoDi/D
    print('Info(D/A) = ' + str(infoDA)) 
    print('------')
 
    IG[ob, 0] = ob+1
    IG[ob, 1] = infoD - infoDA
 
print('IG = ' + str(IG))
IGsorted = IG[IG[:, 1].argsort()]
print('Sortirano IG = ' + str(IGsorted))#sortiran IG 
