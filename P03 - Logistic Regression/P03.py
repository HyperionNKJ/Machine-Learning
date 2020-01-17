import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(df):
    
    N = df.shape[0] # the number of samples
    D = df.shape[1] - 1 # the number of features, excluding a label
    
    X_no_bias = df.values[:, 1:]
    X = np.insert(X_no_bias, D, 1, axis = 1)
    y = df.values[:, 0]
    features = np.array(df.columns[1:])
        
    assert X.shape == (N, D+1) and y.shape == (N, )
    
    return (X, y, features)

def sigmoid(x):
    ret = 1 / (1 + np.exp(1)**-x)
    return ret


class LogisticRegressionSGD():
    
    def __init__(self, lr=0.8, iterations=100000, number_of_ensemble=1000):
        self.lr = lr
        self.iterations = iterations
        self.number_of_ensemble = number_of_ensemble
    
    
    def initialize_w(self, D):
        w = np.zeros(D, )
        assert w.shape == (D, )
        return w
        
        
    def predict(self, X, w):
        scores = X.dot(w.T)
        probs = sigmoid(scores)
        pred = self.label(probs)
        return pred
    
    
    def label(self, probs):
        initial_shape = probs.shape
        probs = probs.flatten()
        classify = lambda prob: 1 if prob > 0.5 else 0
        pred = np.array([classify(prob) for prob in probs])
        return pred.reshape(initial_shape)
    
    
    def get_train_sample(self, X, y, i):
        N = X.shape[0]
        D = X.shape[1]
        
        X_ = X[i%N].reshape(1, D)
        y_ = y[i%N].reshape(1,)
        
        assert X_.shape == (1, D) and y_.shape == (1,)
        return X_, y_
        
        
    def get_loss(self, X, y, w):
        prediction = self.predict(X, w)
        error = (y.reshape(prediction.shape) - prediction) ** 2
        return error
    
    
    def fit(self, X, y):

        N = X['train'].shape[0]
        D = X['train'].shape[1]
        
        self.avg_loss_over_itr = []
        self.test_error_over_itr = []
        self.w_ensemble = self.initialize_w(D)
        
        w = self.initialize_w(D)
        tot_loss = 0
        
        for i in range(self.iterations):
            X_, y_ = self.get_train_sample(X['train'], y['train'], i)
            tot_loss +=  self.get_loss(X_, y_, w)[0]
            
            for j in range(D):
                p_j = -(X_[0][j] * (y_ - sigmoid(X_.dot(w.T))))
                w[j] -= self.lr * p_j

            if (i + 1) % 100 == 0:
                avg_loss = tot_loss / (i + 1)
                test_error = self.get_loss(X['test'], y['test'], w)
                test_error = np.sum(test_error) / len(X['test'])
                self.avg_loss_over_itr.append(avg_loss)
                self.test_error_over_itr.append(test_error)

            if self.iterations - (i+1) < self.number_of_ensemble:
                self.w_ensemble += w / self.number_of_ensemble

        return w
    
    
    def get_accuracy(self, X, y, w):
        difference = self.get_loss(X, y, w)
        accuracy = 1 - np.count_nonzero(difference) / len(X)
        return accuracy
        
        
def get_indices_of_fields(fields, features):
    indices = []
    for field in fields:
        for i in range(0, len(features)):
            if field == features[i]:
                indices.append(i)
    assert len(indices) == len(fields)
    return indices
