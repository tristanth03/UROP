"""
Aut

It interacts with models.jl to access various neural network architectures and utilizes 
the structure defined in NN.jl.
"""

# home.jl
include("NN.jl")
include("models.jl")

# ---------- CONSTANTS ---------- #
MODEL  = model_3LS()
OPT    = "ADAM"        # OPT can be "GD" or "ADAM"... for now ;)
LR     = 0.001
EPOCHS = 1

# ----------  ---------- #
elapsed_time = @elapsed begin
    myNN = NN(MODEL, OPT, LR)
    #train(myNN,EPOCHS)
    loss_history = train(myNN, EPOCHS)
end

# Print results
println("Elapsed time: $elapsed_time seconds")
println("Final Loss: ", loss_history[end])
println("Accuracy: ", accuracy(myNN))

# Plot the loss over time
"""
plot(1:EPOCHS, loss_history, xlabel="Epochs", ylabel="Loss", label="Training Loss", legend=:topleft)
title!("Loss over epoch, [3LS]")
savefig("loss_new_model.png")
"""

# To save the model into a JLD2 file
"""
save_model(myNN,"filename")
"""

println('\n')

# Calculate kernel
elapsed_time = @elapsed begin
    K = kernel(myNN,10000)
end