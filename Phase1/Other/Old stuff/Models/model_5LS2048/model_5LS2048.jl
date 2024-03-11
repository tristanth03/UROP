using Images
using MLDatasets
using Flux
using BSON
using Random


elapsed_time = @elapsed begin
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
        Y_training = Flux.onehotbatch(Y_training, 0:9)
        Y_testing = Flux.onehotbatch(Y_testing, 0:9)

        return X_training, Y_training, X_testing, Y_testing
    end

#--------- Functions
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
            Dense(2048,10,sigmoid) # Hidden Layer 4 -> Output Layer
            )

        param_5LS2048 = Flux.params(m_5LS2048) # The parameters

        return m_5LS2048,param_5LS2048
    end

    function loss_of(model_5LS2048)
        """
        For a loss function, we use MSE (mean squared error).
        """
        loss_5LS2048(X_5LS2048, Y_5LS2048) = Flux.Losses.mse(model_5LS2048(X_5LS2048), Y_5LS2048)

        return loss_5LS2048
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

    #--------- Main Code

    # Data
    X_training, Y_training, X_testing, Y_testing = load_MNIST()

    # Inputs
    lr = 0.0001                     # learning rate
    opt = Adam(lr)            # optimizer
    m_5LS2048, params_5LS2048 = model_5LS2048()
    loss_5LS2048 = loss_of(m_5LS2048)
    epochs = 1
    print_gap = 1            # The step between process prints


    # Training
    train_batch(X_training, Y_training, loss_5LS2048, opt, params_5LS2048, epochs, print_gap)
    loss_update = loss_5LS2048(X_training, Y_training)
    println("Loss update: $loss_update")
end

println("Elapsed time: $elapsed_time seconds")












