
# Assuming that packages are already installed:
# using Pkg
# Pkg.add("Images") ...

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

#--------- Functions
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

function model_5LS2048()
    """
    A 5-layer model using 2048 nodes in the inner layers.
    Using the sigmoid activation function.
    """

    
    m_5LS2048 = Chain(
        Dense(28*28,2048,sigmoid), # Input Layer -> Hidden Layer 1
        Dense(2048,2048,sigmoid), # Hidden Layer 1 -> Hidden Layer 2
        Dense(2048,2048,sigmoid), # Hidden Layer 2 -> Hidden Layer 3
        Dense(2048,2048,sigmoid), # Hidden Layer 3 -> Hidden Layer 4
        Dense(2048,2048,sigmoid), # Hiddden Layer 4 -> Hidden Layer 5
        Dense(2048,10,sigmoid) # Hidden Layer 5 -> Output Layer
        )

    param_5LS2048 = Flux.params(m_5LS2048) # The parameters

    return m_5LS2048,param_5LS2048
end


