"""
Author: Axel Bjarkar Sigurjónsson

It interacts with models.jl to access various neural network architectures and utilizes 
the structure defined in NN.jl.
"""

include("NN.jl")
include("models.jl")

# ---------- CONSTANTS ---------- #
MODEL  = model_3LS()
OPT    = "ADAM"        # OPT can be "GD" or "ADAM"... for now ;)
LR     = 0.01
EPOCHS = 400

# ---------- oooOOOooo ---------- #
elapsed_time = @elapsed begin

    myNN = NN(MODEL,OPT,LR)
    train(myNN,EPOCHS)

end # stopwatch stops
println("Elapsed time: $elapsed_time seconds")
println("Loss: ",get_loss(myNN))
println("Accuracy: ",accuracy(myNN))

# to save model as bson file:
# @save "filename.bson" myNN