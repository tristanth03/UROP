
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

function model_3LS()
    """
    A 3-layer model using 60 nodes in the inner layers.
    Using the sigmoid activation function.
    """

    
    m_3LS = Chain(
        Dense(28*28,60,sigmoid), # Input Layer -> Hidden Layer 1
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

function train_batch(X_train, Y_train, loss, model, opt, params, epochs,print_gap)
    """
    In: data, loss, optimizer, parameters, iteration(epochs)
    Out: trained model
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

opt = Descent(lr) # optimizer
m_3LS,params_3LS = model_3LS()
loss_3LS = loss_of(m_3LS)
epochs = 10
lr = 0.1 # learning rate
print_gap = 2 # The step between process prints

# Training

m_3LS_trained = train_batch(X_training, Y_training, loss_3LS, m_3LS, opt, params_3LS, epochs, print_gap)
loss_update = loss_3LS(X_training,Y_training)
println("Loss update: $loss_update")

# Save parameters
params_dict = Dict("params_3LS" => params_3LS)
save("model_params.jld2", params_dict)