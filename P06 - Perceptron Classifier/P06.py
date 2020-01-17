import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_samples(X_train, y_train):
    fig = plt.figure(figsize=(16,7))
    
    for i in range(30):
        fig.add_subplot(5, 6, i + 1)
        plt.imshow(X_train[i][:-1].reshape(28,28))
    plt.show()
    
    return fig

def sgn(x):
    return (x >= 0)*2-1

# it is to predict accuracy of X data.
def predict(X, y, w):
    accuracy = (1 - np.count_nonzero(y-sgn(X.dot(w))) / X.shape[0]) * 100
    return accuracy

# return w, number_of_misclassifications, test_accuracy
def perceptron(X, y, w, epoch):
   
    number_of_misclassifications = []
    test_accuracy = []
    num_samples = X['train'].shape[0]

    for i in range(epoch):
        number_of_misclassification = 0
        for j in range(num_samples):
            sample = X['train'][j]
            pred = sgn(sample.dot(w))
            truth = y['train'][j]
            if not (pred == truth):
                w += truth * sample
                number_of_misclassification += 1
        number_of_misclassifications.append(number_of_misclassification)
        test_accuracy.append(predict(X['test'], y['test'], w))
    
    return w, number_of_misclassifications, test_accuracy

# plot number_of_misclassifications returned by perceptron
def plot_number_of_misclassifications_over_epochs(errors):
    
    fig = plt.figure(figsize=(17,5))
    plt.plot(range(len(errors)), errors)
    plt.title('Number of missclassifications over epochs', {'fontsize': 20})
    plt.ylabel('Number of missclassifications', {'fontsize': 15})
    plt.xlabel('Epochs', {'fontsize': 15})
    
    return fig

# plot test_accuracy returned by perceptron
def plot_accuracy_over_epochs(test_accuracy):
    
    fig = plt.figure(figsize=(17,5))
    plt.plot(range(len(test_accuracy)), test_accuracy)
    plt.title('Accuracy over epochs', {'fontsize': 20})
    plt.ylabel('Accuracy', {'fontsize': 15})
    plt.xlabel('Epochs', {'fontsize': 15})
    
    return fig