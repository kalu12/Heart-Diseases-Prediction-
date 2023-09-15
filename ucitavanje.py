import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt





data=pd.read_csv('heart.csv') #ucitavanje baze podataka
#premestanje kolone sa  klasom na kraj 
data = data.reindex(['SkinCancer','BMI','Smoking','AlcoholDrinking','Stroke','PhysicalHealth','MentalHealth','DiffWalking','Sex','AgeCategory','Race','Diabetic','PhysicalActivity','GenHealth','SleepTime','Asthma','KidneyDisease','HeartDisease'], axis=1) 
#dodeljivanje numerickih vrednosti kateforickim atributima

data.info(verbose=True)
inform = data.describe().T



data =  data[data.columns].replace({'Excellent':4,'Fair':1,'Good':2,'Very good':3,'White':0,'Black':1,'Asian':2,'Hispanic':3,'Other':4,'Poor':0,'American Indian/Alaskan Native':2,'18-24':0,'25-29':1,'30-34':2,'35-39':3,'40-44':4,'45-49':5,'50-54':6,'55-59':7,'60-64':8,'65-69':9,'70-74':10,'75-79':11,'80 or older':12,'Yes':1, 'No':0, 'Male':1,'Female':0,'No, borderline diabetes':'0','Yes (during pregnancy)':'1' })
data['Diabetic'] = data['Diabetic'].astype(int)




cols = data.columns
suma = pd.DataFrame(index = cols)
for i in cols:
    suma[i] =len(data.loc[data[i]==0, i])
    if (suma[i][0]>len(data)*0.3 and (i=='SleepTime' or i=='BMI')): #ako ima previse nultih vrednosti odbaci kolone
        data.drop(i, axis=1,inplace=True)
        
        
data.info(verbose=True)
inform = data.describe().T


#print(data.isnull().sum())

