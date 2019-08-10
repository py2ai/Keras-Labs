from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import plot_model


model = Sequential([Dense(32, input_shape=(100,)),Activation('relu'),Dense(1), Activation('softmax')])
plot_model(model,to_file='mymodel.png',show_shapes=True)
