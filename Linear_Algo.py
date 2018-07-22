import csv
import matplotlib.pyplot as plt
import numpy as np




def coVariance(meanX , meanY):			# Co-variance = sum((x(i) - mean(X)) * (y(i) - mean(Y)))/ n
	coVar = 0
	counter=0
	for i in X_train_data:
		coVar += (i[0] - meanX) * (Y_train_data[counter] - meanY)
		counter += 1
	
	return coVar/float(X_train_data.shape[0])




def variance(data):
	mean_data = np.mean(data)		# Variance = sum((x(i) - mean(X)) ** 2))/ n
	var = 0
	for i in data:
		var += (i[0] - mean_data)**2

	return var/float(data.shape[0])





def Linear_regr_algo():

	# Calculate mean of data
	X_mean = np.mean(X_train_data)
	Y_mean = np.mean(Y_train_data)

	# Calculate variance of data
	X_var = variance(X_train_data)

	# Calculate co-variance of data
	XY_coVar = coVariance(X_mean,Y_mean)

	# For line y = mx + c , for best fit line : m = Co-variance(X,Y)/Variance(X) , c = mean(Y) - a*mean(X)
	slope = XY_coVar/float(X_var)
	intercept = Y_mean - slope * X_mean

	# Predict data from above best fit line
	Y_predict = []

	for i in X_test_data:
		Y_predict.append((slope * i[0] + intercept))

	# Find the mean squared error
	square_error = 0

	for i in range(Y_test_data.shape[0]):
		square_error += (Y_test_data[i]-Y_predict[i])**2

	mean_error = square_error/float(X_test_data.shape[0])

	print ("Mean squared error by our algorithm is: " , mean_error)

	# Compare with mean squared error with sklean Linear regression model
	sklearn_error_check()

	plt.scatter(X_test_data,Y_test_data, color = "Black")		# Plot data on graph
	plt.plot(X_test_data,Y_predict, color="Brown")

	plt.xticks(())
	plt.yticks(())
	plt.show()		# Show plot




def sklearn_error_check():
	from sklearn import linear_model
	from sklearn.metrics import mean_squared_error
	# Create Linear regression object
	regr = linear_model.LinearRegression()

	# Fit the data
	regr.fit(X_train_data, Y_train_data)

	# Predict the data
	Y_predict=regr.predict(X_test_data)

	# Check for mean squared error
	print("Mean squared error by Sklearn: ",mean_squared_error(Y_predict,Y_test_data))





# Read csv file and extract required feature and label
with open("data.csv") as csvFile:
	reader = csv.reader(csvFile,delimiter=",")		# Dataset with 60 examples
	feature_data=[]
	label_data=[]
	for line in reader:
		# Feature :  (A11)The number of families with an income less than $3000
		feature_data.append(line[11])		# Considering 11th feature for simple Linear Regression
		label_data.append(line[-1])		# Last column represents label



# Convert data to a numpy array
feature_data = np.array(feature_data,dtype='float32')
label_data = np.array(label_data,dtype='float32')

# Extract training and testing data
X_train_data = feature_data[:-10,np.newaxis]
X_test_data = feature_data[-10:,np.newaxis]
Y_train_data = label_data [:-10]
Y_test_data = label_data [-10:]


# Call our Linear Regression algorithm
Linear_regr_algo()
