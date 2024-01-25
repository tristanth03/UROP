"""
Author: Axel Bjarkar Sigurjónsson

It interacts with models.jl to access various neural network architectures and utilizes 
the structure defined in NN.jl.
"""

# home.jl
include("NN.jl")
include("models.jl")

using Plots
using Images
using MLDatasets
using BSON
using FileIO
using Flux
using ImageShow
using ImageInTerminal
using ImageIO
using ImageMagick
using LinearAlgebra
using JLD2

# ---------- CONSTANTS ---------- #
MODEL  = model_3LR_SM()
OPT    = "ADAM"        # OPT can be "GD" or "ADAM"... for now ;)
LR     = 0.001
EPOCHS = 100

# ----------  ---------- #
elapsed_time = @elapsed begin
    myNN = NN(MODEL, OPT, LR)
    train(myNN,EPOCHS)
    # loss_history = train(myNN, EPOCHS)
end

# Plot the loss over time
# plot(1:EPOCHS, loss_history, xlabel="Epochs", ylabel="Loss", label="Training Loss", legend=:topleft)
# title!("Loss over epoch, [New model]")
# savefig("loss_new_model.png")

# # Print results
println("Elapsed time: $elapsed_time seconds")
println("Final Loss: ", loss_history[end])
println("Accuracy: ", accuracy(myNN))
# to save model as bson file:
# @save "filename.bson" model=myNN.model