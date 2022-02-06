import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
#from utils.misc_tools import error_ellipse
#from misc_tools import error_ellipse
#from matplotlib.patches import Ellipse


def plot_single_graph(X_Y, title, xlabel, ylabel, label, is_scatter=False, sigma=None):
    """
    That function plots a single graph

    Args:
        X_Y (np.ndarray): array of values X and Y, array shape [N, 2]
        title (str): sets figure title
        xlabel (str): sets xlabel value
        ylabel (str): sets ylabel value
        label (str): sets legend's label value
    """
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if is_scatter:
        plt.scatter(np.arange(X_Y.shape[0]), X_Y, s=1, label=label, c="b")
        if sigma is not None:
            plt.plot(np.arange(X_Y.shape[0]), sigma, c="orange")
            plt.plot(np.arange(X_Y.shape[0]), -sigma, c="orange")
    elif len(X_Y.shape) ==1:
        plt.plot(np.arange(X_Y.shape[0]), X_Y, label=label)
    else:
        plt.plot(X_Y[:, 0], X_Y[:, 1], label=label)
    
    plt.legend()


def plot_graph_and_scatter(X_Y0, X_Y1, title, xlabel, ylabel, label0, label1, color0='b', color1='r', point_size=1):
    """
    That function plots two graphs, plot and scatter

    Args:
        X_Y0 (np.ndarray): array of values X and Y, array shape [N, 2] of graph 0
        X_Y1 (np.ndarray): array of values X and Y, array shape [N, 2] of graph 1
        title (str): sets figure title
        xlabel (str): sets xlabel value
        ylabel (str): sets ylabel value
        label0 (str): sets legend's label value of graph 0
        label1 (str): sets legend's label value of graph 1
        color0 (str): color of graph0
        color1 (str): color of graph1
        point_size(float): size of scatter points
    """
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(X_Y0[:, 0], X_Y0[:, 1], label=label0, c=color0)
    plt.scatter(X_Y1[:, 0], X_Y1[:, 1], label=label1, s=point_size, c=color1)
    plt.legend()


def plot_four_graphs(X_values, Y0, Y1, Y2, Y3, title, xlabel, ylabel, label0, label1, label2, label3):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(X_values, Y0, label=label0)
    plt.plot(X_values, Y1, label=label1)
    plt.plot(X_values, Y2, label=label2)
    plt.plot(X_values, Y3, label=label3)
    plt.legend()


def plot_three_graphs(X_Y0, X_Y1, X_Y2, title, xlabel, ylabel, label0, label1, label2, shapeandcolor0 = 'b-', shapeandcolor1 = 'k-', shapeandcolor2 = 'g:'):
    plt.figure(figsize=(15, 8))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(X_Y0[:, 0], X_Y0[:, 1], '{}'.format(shapeandcolor0), label=label0)
    plt.plot(X_Y1[:, 0], X_Y1[:, 1], '{}'.format(shapeandcolor1), label=label1)
    plt.plot(X_Y2[:, 0], X_Y2[:, 1], '{}'.format(shapeandcolor2), label=label2)
    plt.legend()

def plot_two_graphs_one_double(X_Y0, X_Y1, X_Y2  ,title, xlabel, ylabel, label0, label1):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(X_Y0[:, 0], X_Y0[:, 1], 'b-', label=label0)
    plt.plot(X_Y1[:, 0], X_Y1[:, 1], 'y-', label=label1)
    plt.plot(X_Y2[:, 0], X_Y2[:, 1], 'y-')
    plt.legend()




def show_graphs():
    plt.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib import animation

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    e1 = Ellipse(xy=(0.5, 0.5), width=0.5, height=0.2, angle=60, animated=True)
    e2 = Ellipse(xy=(0.8, 0.8), width=0.5, height=0.2, angle=100, animated=True)
    ax.add_patch(e1)
    ax.add_patch(e2)

    def init():
        return [e1, e2]

    def animate(i):
        e1.angle = e1.angle + 0.5
        e2.angle = e2.angle + 0.5
        return e1, e2

    anim = animation.FuncAnimation(fig, animate, init_func=init, interval=1, blit=True)
    plt.show()