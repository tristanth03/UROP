using Flux
using JLD2

# Assuming you have the model architecture defined in another script
include("model_3LS.jl")

# Create an instance of your model
m_3LS, params_3LS = model_3LS()

# Load the parameters from the saved file
loaded_params = load("C:\\Programming\\Github\\UROP\\NN_v0.0\\params_m_3LS.jld2")

# Apply the loaded parameters to your model
Flux.loadparams!(m_3LS, loaded_params["params_3LS"])


