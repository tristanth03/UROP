# This is the nureal network struct used for the NTK project
# Authors: Axel Bjarkar Sigurjónsson
#          Tristan Þórðarson

### README
""" This file initalizes the NN and features all of its functions
"""


# --------- Packages --------- #
# Make sure they are installed
using Images
using FileIO
using MLDatasets
using Flux
using ImageShow
using ImageInTerminal
using ImageIO
using ImageMagick
using LinearAlgebra
using JLD2

# --------- Struct --------- #
struct NN
    model
    loss
    opt             # optimizer
end

# --------- Functions --------- #
function load_MNIST()
    """
    Loading the MNIST dataset.
    10 classes of digits from 0 to 9,
    each with 28x28 pixel dimensions.
    X: Grayscale vector, Y: Correct label.
    """

    X_training, Y_training = MNIST(split = :train)[:] 
    X_testing, Y_testing = MNIST(split = :test)[:]

    X_training = Flux.flatten(X_training)
    X_testing = Flux.flatten(X_testing)
    Y_training = Flux.onehotbatch(Y_training,0:9)
    Y_testing = Flux.onehotbatch(Y_testing,0:9)

    return X_training,Y_training,X_testing,Y_testing
end

function loading_bar(epochs)
    """
    Function creates a loading bar for the training process
        
    ATTENTION: this feature might slow down the traing process alot!
    """
    return 1
end

function train(nn::NN, epochs::Int, input_params::String)
    # Load your training data (X_LS3, Y_LS3) here if not already loaded

    # Extract model, loss, and optimizer from the NN object
    model = nn.model
    loss = nn.loss
    opt = nn.optimizer

    # Prepare data in Flux format (you might need to adjust this based on your data structure)
    data = [(X_LS3, Y_LS3) for (X_LS3, Y_LS3) in zip(X_LS3_data, Y_LS3_data)]

    # Training loop
    for epoch in 1:epochs
        Flux.train!(loss, Flux.params(model), data, opt)

        # Print or log training progress if needed
        println("Epoch $epoch: Loss = $(Flux.Losses.mse(model(X_LS3), Y_LS3))")
    end

    # Save trained model parameters
    Flux.save(params(model), input_params)
end



"""
DEMO - how to use

myNN = NN(model, loss, opt)
train(myNNNN, 10, "params.jld2")
"""