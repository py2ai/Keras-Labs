import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
from keras.utils import plot_model
import keras
import pandas as pd
pyplot.style.use('ggplot')
Epochs = 0

# prepare data
X = np.array([1, 1, 1, 1])
X2 = np.array([2, 2, 2, 2])
X3 = np.array([3, 3, 3, 3])
X =np.vstack((X,X2))
X =np.vstack((X,X3))

# prepare target class
y = np.array([0,0,1])
y2 = np.array([0,1,0])
y3 = np.array([1,0,0])

y = np.vstack((y,y2))
y = np.vstack((y,y3))
num_classes=3


print(X.shape,y.shape)

# create the model
model = Sequential()

model.add(Dense(50,input_dim=4))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
plot_model(model, to_file='mymodel.png', show_shapes=True)


# # let's plot metrics

for i in range(10):
	Epochs+=100
	# # training
	history = model.fit(X, y, epochs=Epochs, batch_size=1, verbose=2,shuffle=False)
	df = pd.DataFrame.from_dict(history.history)
	df.to_csv(str(Epochs)+'-loss.csv', encoding='utf-8', index=False)
	
	pyplot.plot(history.history['loss'],label='Epochs: '+str(Epochs)+' categorical_crossentropy')
	
	# pyplot.plot(history.history['sparse_categorical_accuracy'],label='sparse_categorical_accuracy')
	pyplot.legend()
	
	pyplot.xlabel('Training Epoch')
	pyplot.ylabel('Loss Value')
	pyplot.legend()
	# pyplot.show()
	pyplot.savefig('Loss-epochs-{}.jpg'.format(Epochs)) 
	pyplot.close()



u=np.array([1,1,1,1])
if u.shape[0]==4:
	u=np.reshape(u,(1,4))
	
out = np.round_(model.predict(u),2)
print("Prediction of ", u, " is: ")
print(out)


out = np.round_(model.predict(X),2)

print("Prediction of ", X, " is: ")
print(out)





# Binary Accuracy: binary_accuracy, acc
# Categorical Accuracy: categorical_accuracy, acc
# Sparse Categorical Accuracy: sparse_categorical_accuracy
# Top k Categorical Accuracy: top_k_categorical_accuracy (requires you specify a k parameter)
