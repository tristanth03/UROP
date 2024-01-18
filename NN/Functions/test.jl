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

lr = 0.1 # learning rate
opt = Descent(lr) # optimizer
m_3LS,params_3LS = model_3LS()
loss_3LS = loss_of(m_3LS)

include("NN.jl")

myNN = NN(m_3LS, loss_3LS, opt)
train(myNN, 1, params_3LS)