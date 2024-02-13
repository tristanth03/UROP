function create_model(Ni, Nh, No, N, activation)
    layers = [] # Initialize an empty array without specifying the type of elements
    push!(layers, Dense(Ni, N, activation)) # First hidden layer with N nodes and specified activation
    for _ in 2:Nh
        push!(layers, Dense(N, N, activation)) # Additional Nh-1 hidden layers
    end
    push!(layers, Dense(N, No)) # Output layer without specifying an activation function (defaults to identity)
    model = Chain(layers...)|>f64 # Create the model from the layers
    return model
end



function params_model(θ,t)
    
    f = θ[1]*t+θ[2]
    global f
    # Loop over the number of hidden layers
    for n in 1:Nh
        
        
        f = θ[2n+1] * f + θ[2n+2]
        
    end
    return f
end







t1 = [1;0]
t2 = [1;2]

Ni = length(t1) # Dimension of input
No = 2 # Dimension of output
Nh = 1 # Number of hidden layers
N = 2 # Number of nodes in each hidden layer
activation = σ # Sigmoid activation function for hidden layers

model = create_model(Ni, Nh, No, N, activation)
θ = Flux.params(model)
f = params_model(θ,t1)

# Display the models
display(model)
display(f)
display(model(t1))
