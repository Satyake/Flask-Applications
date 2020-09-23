import pandas as pd
data=pd.read_csv('BankNote_Authentication.csv')
y=data.iloc[:,-1].values
x=data.iloc[:,[0,1,2,3]].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.3)
from sklearn.ensemble import RandomForestClassifier

RFC=RandomForestClassifier()
RFC.fit(x_train,y_train)
pred=RFC.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(pred,y_test)


#convert the file to pickle
import pickle
pickle_out=open("classifier.pkl","wb")
pickle.dump(RFC,pickle_out)
pickle_out.close()

RFC.predict([[2,3,4,1]])