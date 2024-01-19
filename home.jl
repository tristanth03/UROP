"""
`home.jl` serves as the core module for initializing and training neural network models. 
It interacts with `models.jl` to access various neural network architectures and utilizes 
the structure defined in `NN.jl` for effective model configuration. 
    
This file acts as a centralized hub, streamlining the process of setting up and training 
artificial neural networks while maintaining modularity and extensibility.
"""

include("NN.jl")
include("models.jl")

# ---------- oooOOOooo ---------- #
myNN = NN(model_3LS(),0.1)
train(myNN,10)