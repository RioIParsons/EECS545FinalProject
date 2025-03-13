import utils
import models
import dataset


data = ...
X_train, Y_train, X_test, Y_test = dataset.partition_data(data)
model = models.KalmanFilter()

model.train_model(X_train, X_train)

y, yhat = model.run(X_test, Y_test)

utils.plot_results(y, yhat)