#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import *
import math
def load_data(path):
    f = open(path)
    data = []
    for line in f.readlines():
        arr = []
        lines = line.strip().split("\t")
        for x in lines:
            arr.append(x)
        data.append(arr)
    return data

# load data from the dataset
pre_train=load_data('./ml-100k/u1.base')
pre_val=load_data('./ml-100k/u1.test')
n_users=943
n_item=1682



# In[2]:


#initialize training matrix and validation matrix
R_train=zeros((n_users,n_item))
R_val=zeros((n_users,n_item))
for i in pre_train:
    user_id=int(i[0])-1
    item_id=int(i[1])-1
    rating=float(i[2])
    R_train[user_id][item_id]=rating
for i in pre_val:
    user_id=int(i[0])-1
    item_id=int(i[1])-1
    rating=float(i[2])
    R_val[user_id][item_id]=rating
#initialize parameters
K=20
alpha=0.2
lamda_p=0.02
lamda_q=0.01
max_epoch=200
# create user and item matrices
P_users=random.random((n_users,K))
Q_item=random.random((n_item,K))
R_train=mat(R_train)
R_val=mat(R_val)
P_users=mat(P_users)
Q_item=mat(Q_item)


# In[3]:


# define loss function
def loss(P,Q,R):
    error=0
    sum_of_P=0
    sum_of_Q=0
    n_p=[]
    n_q=[]
    for i in range(n_users):
        n=0
        for j in range(n_item):
            if R[i,j]>0:
                error+=square(R[i,j]-(P[i]*Q[j].T)[0,0])
                n+=1 #calculate total ratings of a certain user i
        n_p.append(n)
        sum_of_P+=square(linalg.norm(P[i]))*n_p[i]

    for j in range(n_item):
        n=0
        for i in range(n_users):
            if R[i,j]>0:
                n+=1
        n_q.append(n)
        sum_of_Q+=square(linalg.norm(Q[j]))*n_q[j]
    error+=lamda_p*sum_of_P+lamda_q*sum_of_Q #normalization
    return error,n_p,n_q
            


# In[4]:


# Mean Absolute Error
def MAE():
    n=0
    error=0
    for i in range(n_users):
        for j in range(n_item):
            if R_val[i,j]>0:
                error+=abs(R_val[i,j]-(P_users[i]*Q_item[j].T)[0,0])
                n+=1
    Loss=error/n
    return Loss
# Root Mean Square Error
def RMSE():
    n=0
    error=0
    for i in range(n_users):
        for j in range(n_item):
            if R_val[i,j]>0:
                error+=square(R_val[i,j]-(P_users[i]*Q_item[j].T)[0,0])
                n+=1
    Loss=sqrt(error/n)
    return Loss


# In[5]:


MAE_val=[]
RMSE_val=[]
L_trainings=[]
for epoch in range(max_epoch):
    # select a sample randomly
    i=random.randint(1,n_users)
    j=random.randint(1,n_item)
    # calculate gradient on this sample
    error=R_train[i,j]-(P_users[i]*Q_item[j].T)[0,0]
    G_p=-1*error*Q_item[j]+lamda_p*P_users[i]
    G_q=-1*error*P_users[i]+lamda_q*Q_item[j]
    # update vectors in P and Q
    P_users[i]-=alpha*G_p
    Q_item[j]-=alpha*G_q
    L_training=loss(P_users,Q_item,R_train)[0]
    L_validation_RMSE=RMSE()
    L_validation_MAE=MAE()
    # exponential decay periodically
    if epoch%5==0:
        alpha/=2
    print('The current loss of validation is: ',L_validation_RMSE,'\tepoch: ',epoch)
    MAE_val.append(L_validation_MAE)
    RMSE_val.append(L_validation_RMSE)
    L_trainings.append(L_training)


# In[12]:


import matplotlib.pyplot as plt
# show in graph
plt.figure(figsize=(18, 6))
plt.plot(L_trainings, color="r", label="Loss on training set")
#plt.plot(RMSE_val, color="b", label="Lvalidation by RMSE")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("The graph of loss value varing with the number of iterations")
plt.show()


# In[13]:


#calculate the final prediction matrix
R_predict=P_users*Q_item.T
print(R_predict)


# In[ ]:




