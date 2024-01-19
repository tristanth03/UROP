"""
This file includes models that the NN.jl struct takes in as a "model" parameter
"""

# ----------- Packages ----------- #
using Images
using MLDatasets
using Flux
using BSON
using Random
# Make sure you have the packages installed.

# ----------- Models ----------- #
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

function model_3LS()
    """
    A 3-layer model using 60 nodes in the inner layers.
    Using the sigmoid activation function.
    """

    m_3LS = Chain(
        Dense(28*28, 60, sigmoid), # Input Layer -> Hidden Layer 1
        Dense(60, 60, sigmoid),     # Hidden Layer 1 -> Hidden Layer 2
        Dense(60, 10, sigmoid)      # Hidden Layer 2 -> Output Layer
    )

    param_3LS = Flux.params(m_3LS) # The parameters

    return m_3LS, param_3LS
end
