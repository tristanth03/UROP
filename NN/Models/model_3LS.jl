
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

function model_3LS()
    """
    A 3-layer model using 60 nodes in the inner layers.
    Using the sigmoid activation function.
    """

    
    m_3LS = Chain(
        Dense(784,60,sigmoid), # Input Layer -> Hidden Layer 1
        Dense(60,60,sigmoid), # Hidden Layer 1 -> Hidden Layer 2
        Dense(60,10,sigmoid) # Hidden Layer 2 -> Output Layer
        )

    param_3LS = Flux.params(m_3LS) # The parameters

    return m_3LS,param_3LS
end

function loss_of(model_3LS)
    """
    For a loss function we use MSE(mean squared error).
    """
    
   
    loss_3LS(X_LS3,Y_LS3) =  Flux.Losses.mse(model_3LS(X_LS3),Y_LS3) 

    return loss_3LS
end

function train_batch(X_train, Y_train, loss, model, opt, params, epochs, print_gap)
    """
    Trains the model as a batch, ever
    """
    
    data = [(X_train, Y_train)]
    for epoch in 1:epochs
        Flux.train!(loss, params, data, opt)
        if epoch % print_gap == 0 
            println("Epoch $epoch of $epochs completed.")
        end
        
    end
end



#--------- Main Code

# Data
X_training,Y_training,X_testing,Y_testing = load_MNIST()

# Inputs
lr = 0.1 # learning rate
opti = Descent(lr) # optimizer
print_gap = 5 # The step between process prints
m_3LS,params_3LS = model_3LS()

# Training
