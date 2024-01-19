"""
README

DEMO - how to use
myNN = NN(model, learning_rate)
train(myNN, 10, "params.jld2")

This is the nureal network struct used for the NTK project
This file initalizes the NN and features all of its functions

Authors: Axel Bjarkar Sigurjónsson and Tristan Þórðarson
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
    lr          # learning rate 
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
    Can be edited later to include other loss models
    """

    loss(X, Y) = Flux.Losses.mse(model(X), Y)
    return loss
end

function train(nn::NN, epochs, input_params)
    """
    Function trains NN model with either custom or "random" parameters
    """

    # Load MNIST data
    X_train, Y_train, _, _ = load_MNIST()
    data = [(X_train, Y_train)]

    # Check if using custom parameters
    if input_params !== nothing
        # Loading parameters
        if isfile(input_params)
            loaded_dict = load(input_params)
            Flux.loadparams!(nn.model, loaded_dict["model_params"])
        end
    else
        # Initalizes with "Random parameters"
        Flux.params(nn.model)
    end

    # Actual training
    for epoch in 1:epochs
        Flux.train!(loss_of(nn.model), Flux.params(nn.model), data, Descent(nn.lr))
    end

    println("Training completed")
    
end

function save_parameters(nn::NN, filename)
    """
    This function saves the current parameters into a jld2 file
    It double checks not to override existing files without permission
    """

    # Check if filename is valid
    filename = check_jld2(filename)
    
    # Check if overriding a file
    if isfile(filename)
        println("The file $filename already exists. Do you want to override it? (y/n): ")
        user_response = lowercase(readline())
        
        if user_response != "y"
            println("Parameters not saved.")
            return # if we get to this point, the function will terminate
        end
    end
    
    # Actually saving
    param_dict = Dict("model_params" => Flux.params(nn.model))
    save(filename, param_dict)
    println("Parameters saved to $filename")
end

function check_jld2(filename)
    """
    This function returns the correct jld2 format for the filename 
    """
    
    # Check if user already put exrension
    # ext: file EXTension
    base, ext = splitext(filename) 
    
    if ext == ".jld2"
        return filename
    else
        return base * ".jld2"
    end
end
