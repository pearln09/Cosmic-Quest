#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD, Adagrad, RMSprop, Adam, Nadam


# In[2]:


dataset=loadtxt("pima-indians-diabetes.csv",delimiter=',')


# In[3]:


X=dataset[:,:-1]
y=dataset[:,-1]


# In[4]:


print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")


# In[5]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[6]:


print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")


# In[7]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[8]:


def create_model(optimizer='adam'):
    model=Sequential()
    model.add(Dense(24,input_shape=(8,),activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# In[9]:


optimizers = {
    'SGD': SGD(learning_rate=0.01, momentum=0.9),
    'Adagrad': Adagrad(learning_rate=0.01),
    'RMSprop': RMSprop(learning_rate=0.001),
    'Adam': Adam(learning_rate=0.001),
    'Nadam': Nadam(learning_rate=0.001)
}


# In[10]:


for opt_name, opt in optimizers.items():
    print(f"Training with {opt_name} optimizer...")
    model = create_model(opt)
    model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"{opt_name} optimizer - Accuracy: {accuracy:.4f}")


# In[ ]:




