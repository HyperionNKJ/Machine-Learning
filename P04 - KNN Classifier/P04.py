import numpy as np
import collections  # it is optional to use collections

# prediction function is to predict label of one sample using k-NN 
def predict(X_train, y_train, one_sample, k):  
    
    Xy_train = np.insert(X_train, X_train.shape[1], y_train, axis = 1) # combine both X & y to preserve data-label order when sorting
    unsorted_dist2NN = np.array([distance(Xy_train[i], one_sample) for i in range(k)]) # initialize first k training samples
    indicies_order = np.argsort(unsorted_dist2NN)
    dist2NN = unsorted_dist2NN[indicies_order] 
    NN = Xy_train[indicies_order]
    
    for i in range(k, X_train.shape[0]):
        distance_to_query = distance(Xy_train[i], one_sample)
        if (distance_to_query < dist2NN[k-1]):
            j = find_index(distance_to_query, dist2NN) # find where(index) in dist2NN to insert x_i 
            NN[j+1:k] = NN[j:k-1] # shift NN and dist2NN rightwards to make space for x_i
            dist2NN[j+1:k] = dist2NN[j:k-1]
            NN[j] = Xy_train[i] # insert x_i
            dist2NN[j] = distance_to_query
    
    prediction = majority(NN[:,-1].flatten())
    
    return prediction

def distance(x_i, x_q):
    diff = x_i[0:-1] - x_q # omit label of x_i in calculation
    distance = diff.dot(diff.T)
    return distance
    
def find_index(value, arr):
    for i in range(len(arr)):
        if (value < arr[i]):
            return i
    return -1    
    
def majority(arr):
    counts = np.bincount(arr.astype(int))
    max_count = counts.max()
    majorities = np.argwhere(counts == max_count).flatten()
    
    # if only 1 digit has the majority, return it as the prediction. Otherwise, select digit among candidates that have smallest distance
    if (len(majorities) == 1): 
        return majorities[0]
    else:
        for label in arr:
            if (label in majorities): return label
    return -1
        
# accuracy function is to return average accuracy for test or validation sets 
def accuracy(X_train, y_train, X_test, y_test, k):  # You can use def prediction above.
    
    prediction = np.array([predict(X_train, y_train, X_test[i], k) for i in range(len(X_test))])
    difference = y_test - prediction 
    acc = (1 - np.count_nonzero(difference) / len(X_test)) * 100 # 0 means correct prediction. Otherwise, wrong prediction
    return acc

# stack_accuracy_over_k is to stack accuracy over k. You can use def accuracy above.         
def stack_accuracy_over_k(X_train, y_train, X_val, y_val):      

    accuracies = [accuracy(X_train, y_train, X_val, y_val, k) for k in range(1, 21)]
    assert len(accuracies) == 20
    return accuracies