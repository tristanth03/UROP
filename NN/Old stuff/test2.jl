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

function model_3LS()
    """
    A 3-layer model using 60 nodes in the inner layers.
    Using the sigmoid activation function.
    """
    m_3LS = Chain(
        Dense(28*28, 60, sigmoid),   # Input Layer -> Hidden Layer 1
        Dense(60, 60, sigmoid),      # Hidden Layer 1 -> Hidden Layer 2
        Dense(60, 10, sigmoid)        # Hidden Layer 2 -> Output Layer
    )

    param_3LS = Flux.params(m_3LS) # The parameters

    return m_3LS, param_3LS
end

function loss_of(model_3LS)
    """
    For a loss function, we use MSE (mean squared error).
    """
    loss_3LS(X_LS3, Y_LS3) = Flux.Losses.mse(model_3LS(X_LS3), Y_LS3)
    return loss_3LS
end

function train(model, loss_function, optimizer, X, Y, epochs, params_file)
    """
    In: model, loss function, optimizer, input data (X, Y), iteration(epochs), parameters file
    Out: trained model with saved parameters
    """
    if isfile(params_file)
        loaded_dict = load(params_file)
        Flux.loadparams!(model, loaded_dict["model_params"])
    end

    data = [(X, Y)]
    for epoch in 1:epochs
        Flux.train!(loss_function, model, data, optimizer)
    end

    println("Training completed")
    return model
end

# Main code
lr = 0.1                          # learning rate
opt = Descent(lr)                 # optimizer
m_3LS, params_3LS = model_3LS()
loss_3LS = loss_of(m_3LS)

X_train, Y_train, _, _ = load_MNIST()
trained_model = train(m_3LS, loss_3LS, opt, X_train, Y_train, 1, "params_m_3LS_epo10.jld2")
