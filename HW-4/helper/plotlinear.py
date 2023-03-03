import matplotlib.pyplot as plt
import numpy as np


def plotDecisionBoundary(X, Y, w):

        ax = plt.axes()
        data = np.concatenate((X, Y.reshape(-1,1)),axis=1)

        x_min, x_max = np.ceil(max(X[:, 0])) + 1, np.floor(min(X[:, 0])) - 1
        y_min, y_max = np.ceil(max(X[:, 1])) + 1, np.floor(min(X[:, 1])) - 1

        x, y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

        z = w[0] + w[1] * x + w[2] * y

        plt.contour(x, y, z, [0], colors='k')
        plt.scatter(data[data[:, -1] == 1, 0],data[data[:, -1] == 1, 1])
        plt.scatter(data[data[:, -1] == 2, 0],data[data[:, -1] == 2, 1])

        ax.set_title("Plot of Data Points " + "(" + "datasetName" + ")")
        ax.set_ylabel('Feature 2 (X2)')
        ax.set_xlabel('Feature 1 (X1)')

        plt.show()
