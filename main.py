# Author: Bobby Bose
# Assignment 2: Linear and Polynomial Regression


from datetime import datetime

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
# PART 1 of Assignment----------------------------------------------------------------------------------------------

    # Wine data features
    wine_features = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", 
                   "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", 
                   "pH", "sulphates", "alcohol"
                    ]

    # Reading in wine data
    wine_data_df = pd.read_csv("datasets/winequality-red.csv", delimiter = ",")

    # For testing
    #now = datetime.now()
    #current_time = now.strftime("%H:%M:%S")
    #print("Current Time =", current_time)
    #print("")

    print("Running Linear Regression on Wine Data")    
    # Running the linear regression over the wine data
    wine_weights, wine_MSE = multiple_linear_regression(wine_data_df, wine_features, "quality")

    # For testing
    #now = datetime.now()
    #current_time = now.strftime("%H:%M:%S")
    #print("Current Time =", current_time)

    # Printing the final weights and MSE
    print("Final Wine Weights: " + str(wine_weights))
    print("Final Wine MSE: " + str(wine_MSE))

# PART 2 of Assignment----------------------------------------------------------------------------------------------
    # Reading in synthetic data
    synthetic_data_df_1 = pd.read_csv("datasets/synthetic-1.csv", delimiter = ",", names = ["x", "y"])
    synthetic_data_df_2 = pd.read_csv("datasets/synthetic-2.csv", delimiter = ",", names = ["x", "y"])

    # Storing the original data for later
    synthetic_data_original = [synthetic_data_df_1, synthetic_data_df_2]

    # Copying the data so it can be modified
    synthetic_dataset_list = [synthetic_data_df_1.copy(), synthetic_data_df_2.copy()]

    # Preprocessing input features
    for dataset in synthetic_dataset_list:
        x_min = np.min(dataset["x"].to_numpy())
        x_max = np.max(dataset["x"].to_numpy())

        for index, data in dataset.iterrows():
            data["x"] = (data["x"] - x_min) / (x_max - x_min)

    # n values for the polynomials
    polynomial_values = [2, 3, 5]

    # Parameters for the polynomial regression
    alphas = [0.0025, 0.001]
    weight_mins = [0, 0]
    weight_maxes = [1, 1]
    epochs = [4500, 4500]

    # To store regression results
    final_weights = []
    final_weights.append([])
    final_weights.append([])
    final_MSEs = []

    # Running polynomial regression on both datasets for all n values
    for index in range(2):
        for poly_value in polynomial_values:
            print("\nSynthetic " + str(index+1) + " Polynomial Regression for n=" + str(poly_value))
            
            weights, MSE = polynomial_regression(synthetic_dataset_list[index], poly_value, alphas[index], weight_mins[index], weight_maxes[index], epochs[index])
            final_weights[index].append(weights.tolist())
            final_MSEs.append(MSE)

            print("Synthetic " + str(index+1) + " Weights: " + str(weights))
            print("Synthetic " + str(index+1) + " MSE: " + str(MSE))
            print()

# PART 3 of Assignment----------------------------------------------------------------------------------------------
#    Used https://aadhil-imam.medium.com/plotting-polynomial-function-in-python-361230a1e400 as reference for plotting polynomials    
    
    # Setting up subplots
    subplot_titles = ["Synthetic Dataset 1", "Synthetic Dataset 2"]
    figure, axs, = plt.subplots(2, 1, figsize=(30, 45))

    # Plotting both datasets and regression results
    for index, ax in enumerate(axs):
        x_values = synthetic_data_original[index]["x"]
        y_values = synthetic_data_original[index]["y"]
        weights = final_weights[index]  

        # Plotting original data
        axs[index].scatter(x_values, y_values)

        # x values to be plotted using results of polynomial regression
        x_linspace = np.linspace(-2, 2, num = 1000)
        
        # Plotting polynomial regression for n=2
        poly_s1_n2 = []
        for x in x_linspace:
            poly_s1_n2.append(weights[0][2] * (x**2) + weights[0][1] * x + weights[0][0])
        ax.plot(x_linspace, poly_s1_n2, c = 'r', label = "n=2")

        # Plotting polynomial regression for n=3
        poly_s1_n3 = []
        for x in x_linspace:
            poly_s1_n3.append(weights[1][3] * (x**3) + weights[1][2] * (x**2) + weights[1][1] * x + weights[1][0])
        ax.plot(x_linspace, poly_s1_n3, c = 'g', label = "n=3")

        # Plotting polynomial regression for n=5
        poly_s1_n5 = []
        for x in x_linspace:
            poly_s1_n5.append(weights[2][5] * (x**5) + weights[2][4] * (x**4) + weights[2][3] * (x**3) + weights[2][2] * (x**2) + weights[2][1] * x + weights[2][0])
        ax.plot(x_linspace, poly_s1_n5, c = 'm', label = "n=5")

        # Parameters for subplot
        ax.axis("tight")
        ax.set_title(subplot_titles[index], fontsize = 20)
        ax.set_xlim([-2, 2])
        ax.legend(fontsize = 15)

        if index == 0:
            ax.set_ylim([-20, 15])
        else:
            ax.set_ylim([-1.5, 1.5])

    # Setting some more parameters for the overall plot
    plt.suptitle("Polynomial Regression at different n-levels", fontsize = 30)
    plt.show()
    figure.savefig("Regression_plots.png")
# main()


# Description: Runs a linear regression over the dataset
# Arguments: Dataset of examples, features of the dataset, class label for the dataset
# Returns: Weights and MSE
def multiple_linear_regression(dataset, features, class_label):
    
    # Randomly initializing the weights
    weights = np.random.uniform(0, 1, len(features) + 1)

    # Parameters
    y_values = dataset[class_label].to_numpy()
    epochs = 800
    alpha = 0.0001
    MSE = 0


    # Adjusting number of times according to epochs values
    for epoch in range(epochs):

        if epoch % 100 == 0:
            print("Epoch: " + str(epoch))
            print("     Weights: " + str(weights))
            print("     MSE: " + str(get_MSE(dataset, weights, y_values)))

        # Updating each weight
        for weight_index in range(weights.size):
            weights[weight_index] -= alpha * total_loss(dataset, weights, y_values, weight_index)

    MSE = get_MSE(dataset, weights, y_values)
    
    return weights, MSE
# multiple_linear_regression()


# Description: Calculates the loss for the regression equation
# Arguments: Dataset of examples, weights for the equation, y values for the equation, index for the weight/feature being updated
# Returns: Loss  
def total_loss(dataset, weights, y_values, weight_index):
    loss = 0

    # Summing the loss from each example
    for index, data in dataset.iterrows():
        # Retrieving and cleaning up the x_values for the equation 
        x_values = data.to_numpy()
        x_values = np.delete(x_values, -1)
        x_values = np.insert(x_values, 0, 1)
        
        # Calculating the loss for this example
        loss += (np.sum(np.multiply(weights, x_values)) - y_values[index]) * x_values[weight_index]

    return (1/len(y_values)) * loss
# total_loss()


# Description: Calculates and returns the MSE
# Arguments: Dataset of examples, weights for the equation, y values for the equation
# Returns: MSE
def get_MSE(dataset, weights, y_values):

    loss = 0

    # Summing the loss from each example
    for index, data in dataset.iterrows():
        # Retrieving and cleaning up the x_values for the equation 
        x_values = data.to_numpy()
        x_values = np.delete(x_values, -1)
        x_values = np.insert(x_values, 0, 1)

        # Calculating the loss for this example
        loss += (np.sum(np.multiply(weights, x_values)) - y_values[index]) ** 2

    # Returning the full MSE
    return (1/len(y_values)) * loss 
# get_MSE


# Description: Runs a polynomial regression for the given dataset using the parameters given
# Arguments: Dataset being used to train, n value for polynomial, alpha, weight, and epoch parameters for regression
# Returns: MSE and weights for regression equation
def polynomial_regression(dataset, n, alpha, weight_min, weight_max, num_epochs):
    # Randomly initializing the weights
    weights = np.random.uniform(weight_min, weight_max, n+1)

    MSE = 0

    # x and y values of the dataset
    x_values = dataset["x"].to_numpy()
    y_values = dataset["y"].to_numpy()

    # Running for the given amount of epochs
    for epoch in range(num_epochs):
        
        # Printing current statistics every 500 epochs
        if epoch % 500 == 0:
            print("    Epoch: " + str(epoch))
            print("         Weights: " + str(weights))
            print("         MSE: " + str(get_polynomial_MSE(x_values, weights, y_values, n)))

        # Updating each weight
        for weight_index in range(weights.size):
            weights[weight_index] -= alpha * polynomial_loss(x_values, weights, y_values, weight_index, n)

    # Calculating the final MSE
    MSE = get_polynomial_MSE(x_values, weights, y_values, n)

    return weights, MSE
# polynomial_regression()


# Description: Calculates the loss for a given weight
# Arguments: x and y values of the dataset, weights for the hypothesis, which weight is being updated, n level for polynomial
# Returns: Loss for the current weight
def polynomial_loss(x_values, weights, y_values, weight_index, n):
    loss = 0

    # Cycling through all examples
    for index in range(len(y_values)):
        hypothesis = 0
        xi = x_values[index]
        yi = y_values[index]

        # Calculating h0(i)
        for order in range(0,n+1):
            # wi * (xi)^order
            hypothesis += weights[order] * (xi ** order)

        # (h0(i) - y(i)) * xji
        loss += (hypothesis - yi) * (xi ** weight_index)

    return (1/len(y_values)) * loss
# polynomial_lose()


# Description: Calculates the MSE for a given dataset using precalculated weights
# Arguments: x and y values of the dataset, weights for the hypothesis, n level for polynomial
# Returns: MSE for the dataset using the hypothesis
def get_polynomial_MSE(x_values, weights, y_values, n):
    loss = 0

    # Cycling through all the examples
    for index in range(len(y_values)):
        hypothesis = 0
        xi = x_values[index]
        yi = y_values[index]

        # Calculating h0(i)
        for order in range(n):
            # wi * (xi)^order
            hypothesis += weights[order] * (xi ** order)

        # (h0(i) - y(i)) ^ 2
        loss += (hypothesis - yi) ** 2

    return (1/len(y_values)) * loss
# get_polynomial_MSE()


main()

