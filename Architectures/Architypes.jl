### Different Architecture types for varrying data
### Author: Axel Bjarkar

# ?model_architype(architype, dimIN, dimOUT, depth, activation, critical_width=Nothing)
# TO SEE ASCII ART OF LAYOUTS ↑↑↑

### ----- IMPORTS
using Flux
using Random
include("DenseNTK.jl")

### --- CLASSIC model_architype()
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
                |
            |   |   |       
            |   |   |   |    
            |   |   |   |   |
            |   |   |   |    
            |   |   |       
                |
                |    
            
            ↑   ↑           ↑
        dimIN   cri     dimOUT 



        "reverse_funnel"

                        |       
                        |    
                    |   |   |
                |   |   |   |
            |   |   |   |   |
                |   |   |   |
                    |   |   |
                        |       
                        |    
                    
            ↑           ↑   ↑
        dimIN         cri   dimOUT 



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
    function model_architype(architype, dimIN, dimOUT, depth, activation, critical_width)
        depth_validation(depth)

        # Get appropriate width
        if architype == "LH1"
            widths = [dimIN, critical_width, dimOUT]
            depth = 3
        elseif architype == "block"
            widths = widths_block(dimIN, dimOUT, depth, critical_width)
        elseif architype == "funnel"
            widths = widths_funnel(dimIN, dimIN, depth, critical_width)
        elseif architype == "reverse_funnel"
            widths = widths_reverse_funnel(dimIN, dimIN, depth, critical_width)
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

    function widths_funnel(dimIN::Int, dimOUT::Int, depth::Int, critical_width::Int)
        widths = zeros(Int, depth)
        width_decrement = (critical_width - dimOUT) / (depth - 1)

        # Set the widths for each layer
        for i in 2:depth
            widths[i] = critical_width - round(Int, width_decrement * (i - 2))
        end

        widths[1] = dimIN
        widths[depth] = dimOUT

        return widths
    end

    function widths_reverse_funnel(dimIN::Int, dimOUT::Int, depth::Int, critical_width::Int)
        widths = zeros(Int, depth)
        width_increment = (critical_width - dimIN) / (depth - 2)

        # Set the widths for each layer
        for i in 1:depth-1
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



### --- SPECIFC AMOUNT OF PARAMETERS
function construct_model(widths, activation)
    # Model construction
    layers = []
    depth = length(widths)

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

### --- GET #θ from random parameters
function is_nθ_ok(architype, dimIN, dimOUT, depth, critical_width, target_params, error_margin = 0.1)
    # temp_model
    temp_model = model_architype(architype, dimIN, dimOUT, depth, identity, critical_width)

    # lambda function
    nθ = (model) -> begin
        p,_ = Flux.destructure(model)
        return length(p)
    end

    relative_error = abs((nθ(temp_model) - target_params) / target_params)
    
    if relative_error < error_margin
        return true, relative_error
    else
        return relative_error
    end
end

### GET refrence from block
function max_width(block)
    return Int(length(Flux.params(block)[4]))
end

function get_depth(block)
    return Int(length(Flux.params(block))/2)+1
end

### ONE HIDDEN LAYER IS JUST block with depth = 3
function block(dimIN, dimOUT, depth, approx_num_params, activations)
    widths = zeros(Int, depth)
    widths[1] = dimIN
    widths[end] = dimOUT
    
    quad_solve(a,b,c) = (-b+sqrt(b^2 - (4*a*c)))/(2*a)

    if depth != 3
        # A B C fundin með algebru
        A = depth-3
        B = dimIN+1+dimOUT+depth-3
        C = dimOUT-approx_num_params

        nodes = Int(round(quad_solve(A,B,C)))
        
        for i = 2:(depth-1)
            widths[i] = nodes
        end
    else
        widths[2] = Int(round((approx_num_params-dimOUT)/(dimIN+dimOUT+1))) # found with algebra
    end

    return construct_model(widths, activations)
end

function estimate_architecture(architype, dimIN, dimOUT, approx_num_params, ref_block, time_out = 100)
    MIN = 1
    MAX = 3
    THRESHOLD = 0.01

    iteration = 0
    ref_width = max_width(ref_block)
    ref_depth = get_depth(ref_block)

    errors = []

    while iteration < time_out
        depth = ref_depth
        critical_width = rand(round(Int,MIN*ref_width):round(Int,MAX*ref_width))

        if is_nθ_ok(architype, dimIN, dimOUT, depth, critical_width, approx_num_params, THRESHOLD)[1] == true
            current_error = is_nθ_ok(architype, dimIN, dimOUT, depth, critical_width, approx_num_params, THRESHOLD)[2]
            println("Depth: $depth")
            println("Critical Width: $critical_width")
            println("Accuracy: $((1-current_error)*100) %")
            return (depth, critical_width)
        else
            iteration += 1
            push!(errors, is_nθ_ok(architype, dimIN, dimOUT, depth, critical_width, approx_num_params, THRESHOLD))
        end
    end

    println("Target: [$(round((1-minimum(errors))*100, digits=3)) - $(round((1-maximum(errors))*100, digits=3))] %\n")
    error("Time-out error")
end



### ----- ERROR HANDLING
function do_activations_match_depth(depth, activation)
    """Error handling, if using multiple activations"""
    if length(activation) != (depth - 2)
        error("$(length(activation)) activations provided, but $(depth - 2) layers activate.\nBeware: identity is used for last layer - DO NOT PROVIDE FOR LAST LAYER")
    end
end

function depth_validation(depth)
    if depth < 3
        error("Depth must be at least 3 to form an ANN.\nBeware that 'depth' refers to ALL layers.")
    end
end