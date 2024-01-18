
#--------- Functions
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
        Dense(2048,2048,sigmoid), # Hiddden Layer 4 -> Hidden Layer 5
        Dense(2048,10,sigmoid) # Hidden Layer 5 -> Output Layer
        )

    param_5LS2048 = Flux.params(m_5LS2048) # The parameters

    return m_5LS2048,param_5LS2048
end


