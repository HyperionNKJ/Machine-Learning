import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# P1
def read_csv_using_pandas(csv_path='exam_scores.csv'):
    data = pd.read_csv(csv_path)
    print(data.shape)
    print(data.head())
    return data

def parse_pd_data(data, fields=['Circuit', 'DataStructure', 'MachineIntelligence']):
    values = [data[fields[0]].values, data[fields[1]].values, data[fields[2]].values]
    return values

def plot_data(values):
    assert len(values) == 3
    assert type(values[0]) == np.ndarray
    figsize = (6, 4)
    title_fontsize = 20
    label_fontsize = 15

    fig = plt.figure(figsize=figsize)
    ax = Axes3D(fig)

    # scatter
    ax.scatter(values[0],values[1],values[2])
    
    # set title. use title_fontsize above.
    ax.set_title('Score Distributions', fontdict={'fontsize':title_fontsize})

    # set labels for each axes. use label_fontsize above.
    label_font = fontdict={'fontsize':label_fontsize}
    ax.set_xlabel('Circuit', label_font)
    ax.set_ylabel('Data Structure', label_font)
    ax.set_zlabel('Machine Intelligence', label_font)

    plt.show()
    return fig


# P2
def prepare_dataset_for_linear_regression(values):

    bias = np.ones(len(values[0]))
    X = np.array([bias, values[0], values[1]]).T
    y = np.array(values[2])

    return X, y


class LinearRegression:

    def __init__(self, lr=0.0001, iterations=100000):
        self.lr = lr
        self.iterations = iterations
        self.average_rss_history = []

    def fit(self, X, y):
        N = len(y)

        # initialize w
        w = np.zeros((3,1)) 
        y = y.reshape(1000,1)
        
        for i in range(self.iterations):
            
            # implement gradient descent
            gradient_vector = 2/N * X.T.dot(X.dot(w) - y)
            w = w - self.lr * gradient_vector
           
            # compute average_rss
            average_rss = 1/N * ((y - X.dot(w)).T.dot(y - X.dot(w)))
            average_rss = average_rss[0][0] # [[4000]] -> 4000 
            self.average_rss_history.append(average_rss)
        
        self.w = w.flatten()

        
    def predict(self, X):
        pred_y = X.dot(self.w.T)
        return pred_y


def plot_average_rss_history(iterations, history):
    figsize = (6,4)
    title_fontsize = 20
    label_fontsize = 15

    # plot rss_avg history over iterations
    fig = plt.figure(figsize=figsize)
    plt.ylim(0,100)
    
    # initialize x & y data
    x = list(range(iterations))
    y = history
    
    # plot
    plt.plot(x, y)
    plt.title('Average RSS over number of iterations', {'fontsize': title_fontsize})
    plt.ylabel('Average RSS', {'fontsize': label_fontsize})
    plt.xlabel('Iterations', {'fontsize': label_fontsize})
    
    return fig


# P3
def plot_data_with_wireframe(values, w, wireframe_color='red'):
    assert len(w) == 3
    title_fontsize = 20
    label_fontsize = 15
    figsize = (6,4)

    def make_meshgrids(x, y, num=10):

        # make meshgrids for 3D plot.
        x_linspace = np.linspace(np.amin(x), np.amax(x), num)
        y_linspace = np.linspace(np.amin(y), np.amax(y), num)
        
        x_grid, y_grid = np.meshgrid(x_linspace, y_linspace)
        return x_grid, y_grid

    x_grid, y_grid = make_meshgrids(values[0], values[1])
    z_grid = w[0] + w[1] * x_grid + w[2] * y_grid

    fig = plt.figure(figsize=figsize)
    ax = Axes3D(fig)

    # scatter
    ax.scatter(values[0],values[1],values[2])
    
    # plot wireframe
    ax.plot_wireframe(x_grid, y_grid, z_grid, color=wireframe_color)
    
    # set title. 
    ax.set_title('Score Distributions', fontdict={'fontsize':title_fontsize})

    # set labels for each axes.
    label_font = fontdict={'fontsize':label_fontsize}
    ax.set_xlabel('Circuit', label_font)
    ax.set_ylabel('Data Structure', label_font)
    ax.set_zlabel('Machine Intelligence', label_font)

    plt.show()
    return fig


def get_closed_form_solution(X, y):
    
    w = np.zeros(X.shape[1])
    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    return w

