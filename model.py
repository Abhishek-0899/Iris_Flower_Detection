#  importing the libraries

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# load the csv file

df=pd.read_csv(r'C:\Users\dilip.dd\Desktop\cv2file\python project\project2\iris.csv')

print(df.head())

# select independent and dependent variable
df.drop('Id',axis=True,inplace=True)
x=df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y=df['Species']

# split the dataset into train and test

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=50)

# feature scaling

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


#  instantiate the model

classifier=RandomForestClassifier()

# fit the model

classifier.fit(x_train,y_train)

#  make a pickel file

pickle.dump(classifier,open('model.pkl','wb'))

