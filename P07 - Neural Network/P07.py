import numpy as np
import matplotlib.pyplot as plt


# Helper function to plot a decision boundary.
def plot_decision_boundary(pred_func, train_data, color):
    # Set min and max values and give it some padding
    x_min, x_max = train_data[:, 0].min() - .5, train_data[:, 0].max() + .5
    y_min, y_max = train_data[:, 1].min() - .5, train_data[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlGn)
    plt.scatter(train_data[:, 0], train_data[:, 1], c=color, cmap=plt.cm.RdYlGn)


def ReLU(x):
    x = np.maximum(0,x)
    return x 


def sigmoid(x):
    x = 1 / (1 + np.exp(-x))
    return x


def softmax(x):
    exp = np.exp(x)
    result = exp / np.sum(exp, axis=1, keepdims=True)
    return result
    

# Helper function for forward propagation
def forward_propagation(model, X):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    
    h1 = X.dot(W1) + b1
    z1 = ReLU(h1)
    h2 = z1.dot(W2) + b2
    z2 = sigmoid(h2)
    h3 = z2.dot(W3) + b3
    y_hat = softmax(h3) 
    
    cache = {'h1': h1, 'z1': z1, 'h2': h2, 'z2': z2, 'h3': h3, 'y_hat': y_hat}
    
    return y_hat, cache


# Helper function to evaluate the total loss on the dataset
def compute_loss(model, X, y):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    
    y_hat, _ = forward_propagation(model, X)
    total_loss = -np.sum(np.log(y_hat.dot(y.T).diagonal()))
    
    return total_loss


# Helper function to predict an output (0 or 1)
def predict(model, X):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    
    # Forward propagation
    y_hat, _ = forward_propagation(model, X)
    prediction = np.argmax(y_hat, axis=1)
    
    return prediction

def gradient_relu(x):
    x[x <= 0] = 0
    x[x >0 ] = 1
    return x

def back_propagation(model, cache, X, y):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    h1, z1, h2, z2, h3, y_hat = cache['h1'], cache['z1'], cache['h2'], cache['z2'], cache['h3'], cache['y_hat']
    
    dh3 = y_hat - y
    dW3 = (z2.T).dot(dh3)
    db3 = np.sum(dh3, axis=0, keepdims=True)
    dh2 = dh3.dot(W3.T) * ((1 - z2) * z2) 
    dW2 = (z1.T).dot(dh2)
    db2 = np.sum(dh2, axis=0, keepdims=True)
    dh1 = dh2.dot(W2.T) * (gradient_relu(z1))
    dW1 = np.dot(X.T, dh1)
    db1 = np.sum(dh1, axis=0)
    
    gradients = dict()
    gradients['dW3'] = dW3
    gradients['db3'] = db3
    gradients['dW2'] = dW2
    gradients['db2'] = db2
    gradients['dW1'] = dW1
    gradients['db1'] = db1
    return gradients


def randn_initialization(nn_input_dim, nn_hdim1, nn_hdim2, nn_output_dim):
    W1 = np.random.randn(nn_input_dim, nn_hdim1)
    b1 = np.zeros((1, nn_hdim1))
    W2 = np.random.randn(nn_hdim1, nn_hdim2)
    b2 = np.zeros((1, nn_hdim2))
    W3 = np.random.randn(nn_hdim2, nn_output_dim)
    b3 = np.zeros((1, nn_output_dim))
    
    return W1, b1, W2, b2, W3, b3


def const_initialization(nn_input_dim, nn_hdim1, nn_hdim2, nn_output_dim):
    # Constant initialization. why problematic? 
    W1 = np.ones((nn_input_dim, nn_hdim1))
    b1 = np.zeros((1, nn_hdim1))
    W2 = np.ones((nn_hdim1, nn_hdim2))
    b2 = np.zeros((1, nn_hdim2))
    W3 = np.ones((nn_hdim2, nn_output_dim))
    b3 = np.zeros((1, nn_output_dim))
 
    return W1, b1, W2, b2, W3, b3


def build_model(X, y, nn_input_dim, nn_hdim1, nn_hdim2, nn_output_dim,
                lr=0.001, epoch=50000, print_loss=False, init_type='randn'):

    # Initialization
    np.random.seed(0)
    if init_type == 'randn':
        W1, b1, W2, b2, W3, b3 = randn_initialization(nn_input_dim, nn_hdim1, nn_hdim2, nn_output_dim)
    elif init_type == 'const':
        W1, b1, W2, b2, W3, b3 = const_initialization(nn_input_dim, nn_hdim1, nn_hdim2, nn_output_dim)

    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}
    training_loss = []
     
    # Full batch gradient descent. 
    for i in range(epoch):
 
        # Forward propagation
        y_hat, cache = forward_propagation(model, X)
        
        # Backpropagation
        gradients = back_propagation(model, cache, X, y)        

        # Parameter update
        W1 -= lr * gradients['dW1']
        b1 -= lr * gradients['db1']
        W2 -= lr * gradients['dW2']
        b2 -= lr * gradients['db2']
        W3 -= lr * gradients['dW3']
        b3 -= lr * gradients['db3']
         
        # Assign new parameters 
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}
        
        # Print the loss.
        if print_loss and (i+1) % 1000 == 0:
            loss = compute_loss(model, X, y)
            print("Loss (iteration %i): %f" %(i+1, loss))
            training_loss.append(loss)
     
    return model, training_loss


