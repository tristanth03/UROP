"""
Authors: Tristan Þórðarson, Axel Bjarkar Sigurjónsson
Aknowledgements: 

This file includes models that the NN.jl struct takes in as a "model" parameter
"""

# ----------- Packages ----------- #
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

    return m_5LS2048
end

function model_3LS()
    """
    A 3-layer model using 60 nodes in the inner layers.
    Using the sigmoid activation function.
    """

    m_3LS = Chain(
        Dense(28*28, 60, sigmoid), # Input Layer -> Hidden Layer 1
        Dense(60, 60, sigmoid), # Hidden Layer 1 -> Hidden Layer 2
        Dense(60, 10, sigmoid), # Hidden Layer 2 -> Output Layer
        softmax      
    )
    return m_3LS
end

function model_3LR_SM()

    m_3LR_SM = Chain(
        Dense(28*28,784,relu),
        BatchNorm(784),
        Dense(784,784, relu),
        BatchNorm(784),
        Dense(784,10,relu),
        softmax
    )
    return m_3LR_SM
    
end