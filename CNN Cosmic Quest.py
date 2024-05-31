#!/usr/bin/env python
# coding: utf-8

# In[208]:


from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[209]:


dataset=loadtxt("pima-indians-diabetes.csv",delimiter=',')


# In[210]:


X=dataset[:,:-1]
y=dataset[:,-1]


# In[211]:


print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")


# In[212]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[213]:


print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")


# In[214]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[215]:


model=Sequential()
model.add(Dense(12,input_shape=(8,),activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


# In[216]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:





# In[217]:


model.fit(X_train,y_train,epochs=150,batch_size=10,verbose=0)


# In[ ]:





# In[218]:


_,accuracy=model.evaluate(X_train,y_train,verbose=0)
print("Accuracy:%2f"%(accuracy*100))


# In[219]:


loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")


# In[ ]:




