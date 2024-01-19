using Flux
using Flux: Descent
using BSON: @load

function loss_of(model_3LS)
    """
    For a loss function, we use MSE (mean squared error).
    """
    loss_3LS(X_LS3, Y_LS3) = Flux.Losses.mse(model_3LS(X_LS3), Y_LS3)
    return loss_3LS
end

function train_batch(X_train, Y_train, loss, opt, params, epochs, print_gap)
    """
    In: data, loss, optimizer, parameters, iteration(epochs)
    Out: trained model with saved parameters
    """

    data = [(X_train, Y_train)]
    for epoch in 1:epochs
        Flux.train!(loss, params, data, opt)

        if epoch % print_gap == 0
            println("Epoch $epoch of $epochs completed.")
        end
    end
end

# Load MNIST data (assuming you have a load_MNIST function)
# Replace this with your actual data loading code
function load_MNIST()
    # Your data loading code here
end

# Model definition for m_3LS
function model_3LS()
    m_3LS = Chain(
        Dense(28*28, 60, sigmoid),  # Input Layer -> Hidden Layer 1
        Dense(60, 60, sigmoid),      # Hidden Layer 1 -> Hidden Layer 2
        Dense(60, 10, sigmoid)       # Hidden Layer 2 -> Output Layer
    )
    params_3LS = Flux.params(m_3LS)
    return m_3LS, params_3LS
end

# Data loading
X_training, Y_training, X_testing, Y_testing = load_MNIST()

# Hyperparameters
lr = 0.01                     # learning rate
opt = Descent(lr)             # optimizer
m_3LS, params_3LS = model_3LS()
loss_3LS = loss_of(m_3LS)
epochs = 1
print_gap = 1               # The step between process prints

# Check if saved model parameters exist and load them
if isfile("model_params.bson")
    @load "model_params.bson" params_3LS
end

# Training
train_batch(X_training, Y_training, loss_3LS, opt, params_3LS, epochs, print_gap)

# Save model parameters
@save "model_params.bson" params_3LS

# Calculate and print loss update
loss_update = loss_3LS(X_training, Y_training)
println("Loss update: $loss_update")
