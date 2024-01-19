"""
Author: Axel Bjarkar Sigurjónsson

home.jl serves as the core module for initializing and training neural network models. 
It interacts with models.jl to access various neural network architectures and utilizes 
the structure defined in NN.jl for effective model configuration. 
    
This file acts as a centralized hub, streamlining the process of setting up and training 
artificial neural networks while maintaining modularity and extensibility.
"""

include("NN.jl")
include("models.jl")

# ---------- CONSTANTS ---------- #
MODEL  = model_3LS()
OPT    = "GD"        # OPT can be "GD" or "ADAM"... for now ;)
LR     = 0.1
EPOCHS = 10

# ---------- oooOOOooo ---------- #
elapsed_time = @elapsed begin

    myNN = NN(MODEL,LR)
    train(myNN,EPOCHS)

end # stopwatch stops
println("Elapsed time: $elapsed_time seconds")