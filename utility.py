import numpy as np
from activation_functions import *

def plot_activation_functions():
    """
    plot_activation_functions
    Plots the activation functions on a graph
    Parameters
    ----------
    None
    
    Returns
    -------
    None 
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    activation_functions = [step_function, linear_function, \
                            ReLU_function, sigmoid_function, tanh_function]
    labels = ['step', 'linear', 'ReLU', 'sigmoid', 'tanh']
    
    fig, ax = plt.subplots(figsize=(12,8))
    sns.set_palette("tab10")
    
    for l, func in enumerate(activation_functions):
        x = np.linspace(-5 ,5, 500)
        y = [func(i, 1, 0) for i in x]
        sns.lineplot(x=x, y=y, label=labels[l], linewidth=5)

    # include leaky ReLU option
       
    plt.grid()
    plt.ylim([-2, 2])
    plt.legend();



def time_comparison(input_size, iterations):
    """
    Compares the time it takes to run each activation function
    
    Parameters
    ----------
    input_size : int
        The size of the input
    iterations : int
        Number of iterations to compare
    
    Returns
    -------
    None 

    """
    from time import time as timer
    
    activation_functions = [step_function, linear_function, \
                            ReLU_function, sigmoid_function, tanh_function]
    labels = ['step', 'linear', 'ReLU', 'sigmoid', 'tanh']
    
    for l, func in enumerate(activation_functions):
        total_time = 0
        for i in range(iterations):
            # sampling low and high ranges to be researched
            sample = np.random.uniform(low=-1, high=1, size=(input_size,))
            weights = np.random.uniform(low=-1, high=1, size=(input_size,))
            bias = np.random.uniform(low=-1, high=1)
            
            t_start = timer()
            func(sample, weights, bias)
            duration = timer() - t_start
            total_time += duration
            
        avg_time =  total_time/iterations
        print(f"Average time for {labels[l]}: {(avg_time*1000) :0.6f} ms")