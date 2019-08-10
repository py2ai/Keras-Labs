from keras.models import Sequential
from keras.utils import plot_model
from keras.layers import Dense
model = Sequential()
model.add(Dense(10,input_shape=(3, 1), activation='sigmoid'))
plot_model(model, to_file='mymodel.png', show_shapes=True)

