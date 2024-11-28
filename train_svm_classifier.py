import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# load data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# initialize the SVM model with RBF kernel and probablity enabled
model = SVC(kernel='rbf', probability=True)

# train the model
model.fit(x_train, y_train)

# make predictions
y_predict = model.predict(x_test)

# calculate the accuracy score
score = accuracy_score(y_predict, y_test)

# output the accuracy
print('{}% of samples were classified correctly!'.format(score * 100))

# save the model
f = open('svm_model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()