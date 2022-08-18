import pickle
import random
import numpy as np
np.random.seed(19980219)  
data = pickle.load(open('data.pkl', 'rb'))
randid = np.random.permutation(len(data[0]))
train_data = []
for i in range(len(data)):
    train_data.append([data[i][randid[j]] for j in range(int(len(randid)*0.9))])
test_data = []
for i in range(len(data)):
    test_data.append([data[i][randid[j]] for j in range(int(len(randid)*0.9), len(randid))])
pickle.dump(train_data, open('traindata.pkl', 'wb'))
pickle.dump(test_data, open('valdata.pkl', 'wb'))