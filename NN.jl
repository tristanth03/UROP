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
using JLD2
using FileIO
using Flux
using ImageShow
using ImageInTerminal
using ImageIO
using ImageMagick
using LinearAlgebra
using Zygote
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

    loss_history = []
    # Actual training
    for epoch in 1:epochs
        Flux.train!(loss_of(nn.model), Flux.params(nn.model), data, opt(nn))
        push!(loss_history,get_loss(nn))
    
    end


    println("Training completed")

    return loss_history
end

function save_model(nn::NN, filename::String)
    """
    This functions saves the NN in its current state into a JLD2 file.

    To install the NN:
    metadata = JLD2.load("model_info.jld2", "metadata")
    rebuild = JLD2.load("model_info.jld2", "rebuild")

    model = rebuild(metadata)

    Beware: The model is not a type::NN, so it does not support the functions of the NN struct
            It's a standard Flux model, so it can perform all Flux 
    """

    # Check if filename is valid
    filename = check_ext(filename)
    
    # Check if overriding a file
    if isfile(filename)
        println("\nThe file $filename already exists. Do you want to override it? (y/n): ")
        user_response = lowercase(readline())
        
        if user_response != "y"
            println("Model not saved.")
            return # if we get to this point, the function will terminate
        end
    end
    
    # Actually saving
    metadata,rebuild = Flux.destructure(nn.model)
    JLD2.jldsave(filename, metadata=metadata, rebuild=rebuild)
    println("Model saved to $filename")
end

function check_ext(filename::String)
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

function kernel(nn::NN, n=60000)
    """
    This function computes the "Kernel" of a given NN
    Uses jacobian instead of gradient, because gradient outputs only scalars.
    """
    x = load_MNIST()[1]     # training data
    n = size(x)[2]          # number of data no_points
    K = zeros(Int32, n, n)  # Initialize empty Kernel
    
    gradients = [] # will have n unique gradients
    model = nn.model

    for i in 1:n
        xi = X_batch[:, i] # singular picture vector
        push!(gradients, jacobian(Flux.params(model)) do
            y = model(xi)
            return y
        end)
    end

    for i in 1:n
        A = gradients[i]
        
        for j in 1:n
            B = gradients[j]
            K[i,j] = dot(A,B)
        end
    end

    return K
end

