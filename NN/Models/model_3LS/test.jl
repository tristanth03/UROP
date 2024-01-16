using Flux
using JLD2

# Assuming you have the model architecture defined in another script
include("your_model_script.jl")

# Create an instance of your model
m_3LS, params_3LS = model_3LS()

# Load the parameters from the saved file
loaded_params = load("path/to/model_params.jld2")

# Apply the loaded parameters to your model
Flux.loadparams!(m_3LS, loaded_params["params_3LS"])

# Now, your model (m_3LS) is loaded with the trained parameters
