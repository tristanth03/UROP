using ForwardDiff
using Zygote
include("NN.jl"); include("models.jl")

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

params,rebuild = Flux.destructure(myNN.model)
input_data = load_MNIST()[1]
input_data = input_data[:,1]

jacob = ForwardDiff.jacobian(params -> myNN.model(input_data), params)

