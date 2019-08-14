from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
from keras.utils import plot_model
pyplot.style.use('ggplot')

Epochs = 50
# input data
X = array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
print(X.shape)
#create model
model = Sequential()
model.add(Dense(2, input_shape=(10,)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', 'cosine'])
plot_model(model, to_file='mymodel.png', show_shapes=True)
for i in range(10):
	# train model
	Epochs+=50
	history = model.fit(X, [1], epochs=Epochs, batch_size=len(X), verbose=2)
	# plot metrics
	pyplot.title('Epochs: {}'.format(Epochs))
	pyplot.plot(history.history['mean_squared_error'],label='mse')
	pyplot.plot(history.history['mean_absolute_error'],label='mae')
	pyplot.plot(history.history['mean_absolute_percentage_error'],label='mape')
	pyplot.plot(history.history['cosine_proximity'],label='cosine')
	pyplot.xlabel('Training Epoch')
	pyplot.ylabel('Metric Value')
	pyplot.legend()
	pyplot.savefig('Spring-epochs-{}.jpg'.format(Epochs)) 
	#pyplot.show()
	pyplot.close()
	

