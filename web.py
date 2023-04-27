import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

data = pd.read_csv('iris.csv')
x = data.drop(['Classification'],axis =1)
y = data['Classification'] 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.3,random_state =32)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
#fitting the model
model = model.fit(x_train,y_train)
#Saving the model to disk

pickle.dump(model,open('model.pkl','wb'))
#pickle for converting to byte stream, serialising 
#and deserializing
