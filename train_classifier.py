import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# flatten the data 
data = data.reshape(data.shape[0], -1)

# create training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

# train classifers
model.fit(x_train, y_train)

# create predictions
y_predict = model.predict(x_test)

print(f"Accuracy: {accuracy_score(y_test, y_predict) * 100:.2f}%")

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()