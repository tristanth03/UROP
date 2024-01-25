"""

Authors: Axel Bjarkar Sigurjónsson and Tristan Þórðarson
Aknowledgements: 


README

DEMO - how to use
myNN = NN(model, learning_rate)
train(myNN, 10, "params.jld2")

This is the nureal network struct used for the NTK project
This file initalizes the NN and features all of its functions


"""

# ----------- Packages ----------- #
using Images
using MLDatasets
using BSON
using FileIO
using Flux
using ImageShow
using ImageInTerminal
using ImageIO
using ImageMagick
using LinearAlgebra

# --------- Struct --------- #
struct NN
    model::Any
    opt         # optimatzation method, so far only GD and ADAM
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

function get_loss(nn::NN)
    """
    Evaluates loss function
    """

    X_training, Y_training, _, _ = load_MNIST()
    model = nn.model
    loss_fn = loss_of(model)
    loss_update = loss_fn(X_training, Y_training)
    return loss_update
end

function opt(nn::NN)
    """
    Function return the appropriate optimatzation method
    """

    if nn.opt == "GD"
        return Descent(nn.lr)
    elseif nn.opt == "ADAM"
        return Adam(nn.lr)
    end
end

function train(nn::NN, epochs, input_params=nothing)
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

    # loss_history = []
    # Actual training
    for epoch in 1:epochs
        Flux.train!(loss_of(nn.model), Flux.params(nn.model), data, opt(nn))
        # push!(loss_history,get_loss(nn))
    
    end


    println("Training completed")

    return loss_history
end

function save_parameters(nn::NN, filename)
    """
    This function saves the current parameters into a jld2 file
    It double checks not to override existing files without permission
    """

    # Check if filename is valid
    filename = check_ext(filename)
    
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

function check_ext(filename)
    """
    This function returns the correct jld2 format for the filename 
    """
    
    # Check if user already put exrension
    # ext: file EXTension
    base, ext = splitext(filename) 
    if filetype == "jld2"
        if ext == ".jld2"
            return filename
        else
            return base * ".jld2"
        end
    end
    # can change to include bson files
end

function accuracy(nn::NN)
    """
    Evaluates the accuracy of the neural network on the MNIST testing set
    """

    X_testing, Y_testing = load_MNIST()[3:4]
    model = nn.model

    Y_pred = argmax(model(X_testing), dims=1)
    Y_true = argmax(Y_testing, dims=1)

    correct_predictions = sum(Y_pred .== Y_true)
    total_samples = size(Y_testing)[2]
    
    acc = correct_predictions / total_samples
    return acc
end