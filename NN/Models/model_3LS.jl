
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

function model_3LS(scale)
    """
    A 3-layer model using 60 nodes in the inner layers.
    Using the sigmoid activation function.
    """

    
    m_3LS = Chain(
        Dense(scale*scale,60,sigmoid), # Input Layer -> Hidden Layer 1
        Dense(60,60,sigmoid), # Hidden Layer 1 -> Hidden Layer 2
        Dense(60,10,sigmoid) # Hidden Layer 2 -> Output Layer
        )

    param_3LS = Flux.params(m_3LS) # The parameters

    return m_3LS,param_3LS
    
end