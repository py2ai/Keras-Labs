from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import plot_model
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import numpy as np


model = Sequential()
model.add(Dense(10, input_shape=(4,)))
model.add(Activation('relu'))
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('relu'))

plot_model(model, to_file='mymodel.png', show_shapes=True)


model.compile(loss='mean_squared_error',optimizer='sgd',metrics=['accuracy'])

# Generate dummy data

sample1 = np.array([1,1,1,1])
sample2 = np.array([2,2,2,2])
sample3 = np.array([3,3,3,3])

data = np.vstack((sample1,sample2,sample3))


label1 = np.array([1])
label2 = np.array([2])
label3 = np.array([3])

labels = np.vstack((label1,label2,label3))




# Train the model, iterating on the data in batches of 1 samples
model.fit(data, labels, epochs=10, batch_size=3)
			  


out = model.predict(data)
print("Target is:", labels)
print("Prediction is:", out)
