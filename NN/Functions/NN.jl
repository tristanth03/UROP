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
    model::Any
    lr
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
    Y_training = Flux.onehotbatch(Y_training, 0:9)
    Y_testing = Flux.onehotbatch(Y_testing, 0:9)
    return X_training, Y_training, X_testing, Y_testing
end

function loss_of(model)
    """
    For a loss function, we use MSE (mean squared error).
    """

    loss(X, Y) = Flux.Losses.mse(model(X), Y)
    return loss
end

function train(nn::NN, epochs, input_params)
    """
    In: NN struct, iteration(epochs), input_params
    Out: trained model with saved parameters
    """

    # Loss
    loss_fn = loss_of(nn.model)

    # Load MNIST data
    X_train, Y_train, _, _ = load_MNIST()

    if isfile(input_params)
        loaded_dict = load(input_params)
        Flux.loadparams!(nn.model, loaded_dict["model_params"])
    end

    data = [(X_train, Y_train)]

    for epoch in 1:epochs
        Flux.train!(loss_fn, Flux.params(nn.model), data, Descent(nn.lr))
    end

    println("Training completed")
end

# DEMO - how to use
# myNN = NN(model, loss, opt)
# train(myNN, 10, "params.jld2")
