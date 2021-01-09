from sklearn.linear_model import LogisticRegression

#Function to train model
def train(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

#Fuction for prediction
def predict(model, X_test):
    return model.predict(X_test)
