from sklearn.model_selection import train_test_split
X = data.drop('class', axis = 1)
Y = data['class']
x_train, x_test, y_train, y_test = train_test_split(X,Y)
x_train = x_train.values
x_test = x_test.values
