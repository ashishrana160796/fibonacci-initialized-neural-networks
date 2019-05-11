# An array surface plotting the dimensions and values of array for better visualization explaining the distribution of array values.
# A 3-D Plotter function for 2-D 'W' weight array with its dimensions as X and Y axis.


# import statements for 3-D Weight Matrix Surface Plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import random


def plot_3d(X_val, Y_val, W):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(0, X_val,1)
    Y = np.arange(0, Y_val,1)
    X_dash = X
    Y_dash = Y
    X, Y = np.meshgrid(Y, X)

    Z = np.zeros((X_val, Y_val))
    for i in X_dash:
        for j in Y_dash:
                Z[i][j] = W[i][j]
            
    # Axis Labelling
    ax.set_xlabel('X-Index', fontsize=13)
    ax.set_ylabel('Y-Index', fontsize=13)
    ax.set_zlabel('Value', fontsize=13)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_zlim(-4.5, 4.5)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # show plot
    plt.show()
    
    # save plot in given directory
    # fig.savefig('weight_array_plot.png')


def fib_init(X_val,Y_val):

    mul_fact = 0.001
    
    fib_series = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]
    
    W = np.zeros((X_val, Y_val))

    for i in range(X_val):
        for j in range(Y_val):
            W[i][j] = random.choice(fib_series) * mul_fact
            if(random.uniform(0, 1)<0.5):
                W[i][j] = -W[i][j]
    
    return W
    
def plot_hist(W):

    # for saving the figure.
    # fig = plt.figure()

    plt.hist(W, density=True, bins=8)
    plt.ylabel('Count Distribution', fontsize=13)
    plt.xlabel('Array Values', fontsize=13)
    plt.show()
    
    # for saving the figure.
    # fig.savefig('weight_dist_plot.png')
    
def main():
    # W =  np.random.randn(784,300)
    
    W = fib_init(784,300)
    # print(W)
    plot_3d(784,300,W)
    plot_hist(W)


if __name__ == '__main__':
    main()
