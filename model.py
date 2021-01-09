from sklearn.linear_model import LogisticRegression


def train(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    return model.predict(X_test)
