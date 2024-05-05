### Different Architecture types for varrying data
### Author: Axel Bjarkar

# ?model_architype(architype, dimIN, dimOUT, depth, activation, critical_width=Nothing)
# TO SEE ASCII ART OF LAYOUTS ↑↑↑

### ----- IMPORTS
using Flux
include("DenseNTK.jl")

### ----- FUNCTIONS
"""
    NOTE: depth is ALL LAYERS (input, hidden and output)
    NOTE: critical_width is the width of the hidden layer that is critical for the model to work
    
    Model architypes (simplified ASCII art):

    "LH1" - for one hidden layer (so depth=3 for this case)
    
    <---------depth--------->

                |            
    |           |           |
    |  <--σ-->  |           |
    |           |           |
                |            

    ↑           ↑           ↑
    dimIN   critical_width  dimOUT 



    "block"
    
        |   |   |   |   | 
    |   |   |   |   |   |   |
    |   |   |   |   |   |   |
    |   |   |   |   |   |   |   
        |   |   |   |   |
    
    ↑           ↑           ↑
    dimIN   critical_width  dimOUT 



    "funnel"

        |
        |   |
        |   |   |       
        |   |   |   |    
        |   |   |   |   |
        |   |   |   |    
        |   |   |       
        |   |
        |        
        
        ↑               ↑
    dimIN           dimOUT 



    "reverse_funnel"

                        |   
                    |   |
                |   |   |
            |   |   |   |
        |   |   |   |   |
            |   |   |   |
                |   |   |
                    |   |   
                        |
                
        ↑               ↑
      dimIN           dimOUT 



    "hourglass"

        |                           |
        |   |                   |   |
        |   |   |           |   |   |
        |   |   |   |   |   |   |   |
        |   |   |   |   |   |   |   |  
        |   |   |   |   |   |   |   |
        |   |   |           |   |   |    
        |   |                   |   |
        |                           |
        
        ↑           ↑   ↑           ↑
      dimIN    critical_width    dimOUT 



    "diamond"
    
                |   |            
            |   |   |   |        
        |   |   |   |   |   |    
    |   |   |   |   |   |   |   |
    |   |   |   |   |   |   |   |  
    |   |   |   |   |   |   |   |
        |   |   |   |   |   |        
            |   |   |   |        
                |   |            
    
    ↑           ↑   ↑           ↑
    dimIN    critical_width   dimOUT 
"""
function model_architype(architype, dimIN, dimOUT, depth, activation, critical_width=Nothing)
    depth_validation(depth)

    # Get appropriate width
    if architype == "LH1"
        widths = [dimIN, critical_width, dimOUT]
        depth = 3
    elseif architype == "block"
        widths = widths_block(dimIN, dimOUT, depth, critical_width)
    elseif architype == "funnel"
        widths = widths_funnel(dimIN, dimIN, depth)
    elseif architype == "reverse_funnel"
        widths = widths_reverse_funnel(dimIN, dimIN, depth)
    elseif architype == "hourglass"
        widths = widths_hourglass(dimIN, dimOUT, depth, critical_width)
    elseif architype == "diamond"
        widths = widths_diamond(dimIN, dimOUT, depth, critical_width)
    else
        current_types = "Current supported types:\n"
        supported_types = ["LH1", "block", "funnel", "reverse_funnel", "hourglass", "diamond"]
        error("'$architype' is not a valid architecture type\n\n$current_types$(join(supported_types, '\n'))\n")
    end

    # Model construction
    layers = []
    if isa(activation, Function)
        # All layers use the same activation function
        for i in 1:depth-1
            act = i < depth-1 ? activation : identity   # if i < depth-1 use activation else use identity function
            push!(layers, DenseNTK(widths[i], widths[i+1], act))
        end
    elseif isa(activation, Vector)
        # Different activation for each layer
        do_activations_match_depth(depth, activation)
        for i in 1:depth-1
            act = i < depth-1 ? activation[i] : identity
            push!(layers, DenseNTK(widths[i], widths[i+1], act))
        end
    else
        error("Invalid activation type: must be a Function or Vector of Functions.")
    end

    model = Chain(layers...)

    return model
end


### ----- WIDTH FUNCTIONS
function widths_block(dimIN, dimOUT, depth, width)
    widths = ones(Int, depth)*width
    widths[1] = dimIN
    widths[depth] = dimOUT
    
    return widths
end

function widths_funnel(dimIN::Int, dimOUT::Int, depth::Int)
    widths = zeros(Int, depth)
    width_decrement = (dimIN - dimOUT) / (depth - 1)

    # Set the widths for each layer
    for i in 1:depth
        widths[i] = dimIN - round(Int, width_decrement * (i - 1))
    end

    widths[1] = dimIN
    widths[depth] = dimOUT

    return widths
end

function widths_reverse_funnel(dimIN::Int, dimOUT::Int, depth::Int)
    widths = zeros(Int, depth)
    width_increment = (dimOUT - dimIN) / (depth - 1)

    # Set the widths for each layer
    for i in 1:depth
        widths[i] = dimIN + round(Int, width_increment * (i - 1))
    end

    widths[1] = dimIN
    widths[depth] = dimOUT

    return widths
end

function widths_hourglass(dimIN, dimOUT, depth, min_width)
    widths = zeros(Int, depth) # n - Zero Vector
    if depth%2 == 0
        middle_layers = (depth ÷ 2, depth ÷ 2 + 1)
        widths[middle_layers[1]], widths[middle_layers[2]] = min_width, min_width

        # Calculate decrease and increase steps
        decrease_step = (dimIN - min_width) / (middle_layers[1] - 1)
        increase_step = (dimOUT - min_width) / (middle_layers[1] - 1)

        # Set widths for decreasing and increasing phases
        for i in 1:(middle_layers[1] - 1)
            widths[i] = dimIN - round(Int, decrease_step * (i - 1))
        end
        for i in (middle_layers[2] + 1):depth
            widths[i] = min_width + round(Int, increase_step * (i - middle_layers[2]))
        end
    else
        middle_layer = ceil(Int, depth/2)
        widths[middle_layer] = min_width

        # Calculate decreasing widths from the input to the middle layer
        decrease_step = (dimIN - min_width) / (middle_layer - 1)
        for i in 1:(middle_layer-1)
            widths[i] = dimIN - round(Int, decrease_step * (i - 1))
        end

        # Calculate increasing widths from the middle layer to the output
        increase_step = (dimOUT - min_width) / (middle_layer - 1)
        for i in (middle_layer+1):depth
            widths[i] = min_width + round(Int, increase_step * (i - middle_layer))
        end
    end
    widths[depth] = dimOUT

    return widths
end

function widths_diamond(dimIN::Int, dimOUT::Int, depth::Int, max_width::Int)
    widths = zeros(Int, depth)

    if depth % 2 == 1 #ODD
        middle_index = ceil(Int, depth / 2)
        widths[middle_index] = max_width

        # Calculate width increments/decrements
        expand_step = (max_width - dimIN) / (middle_index - 1)
        contract_step = (max_width - dimOUT) / (middle_index - 1)
        
        # Set widths for expansion phase
        for i in 1:(middle_index - 1)
            widths[i] = dimIN + round(Int, expand_step * (i - 1))
        end
        # Set widths for contraction phase
        for i in (middle_index + 1):depth
            widths[i] = max_width - round(Int, contract_step * (i - middle_index))
        end
    
    else #EVEN
        middle_first = depth ÷ 2
        middle_second = middle_first + 1
        widths[middle_first], widths[middle_second] = max_width, max_width

        # Calculate width increments/decrements
        expand_step = (max_width - dimIN) / (middle_first - 1)
        contract_step = (max_width - dimOUT) / (middle_first - 1)

        # Set widths for expansion phase
        for i in 1:(middle_first - 1)
            widths[i] = dimIN + round(Int, expand_step * (i - 1))
        end
        # Set widths for contraction phase
        for i in (middle_second + 1):depth
            widths[i] = max_width - round(Int, contract_step * (i - middle_second))
        end
    end

    widths[1] = dimIN
    widths[depth] = dimOUT

    return widths
end

### ----- ERROR HANDLING
function do_activations_match_depth(depth, activation)
    """Error handling, if using multiple activations"""
    if length(activation) != (depth - 2)
        error("$(length(activation)) activations provided, but $(depth - 2) layers activate.")
    end
end

function depth_validation(depth)
    if depth < 3
        error("Depth must be at least 3 to form an ANN.\nBeware that 'depth' refers to ALL layers.")
    end
end

include("FastNTKMethods.jl")
x = hcat([5 6 7 8 9 10])
m = model_architype("block", 1, 10, 5+2,σ,300)

@time K = kernel(m,x,true,4)