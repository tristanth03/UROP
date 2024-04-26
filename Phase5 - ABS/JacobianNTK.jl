using Flux, ReverseDiff, LinearAlgebra, ProgressMeter

function check_dim(x)
    """This function checks the appropriate  dimensions of input data"""
    if isa(x, Matrix)
        return size(x, 2)  # Returns the number of columns (width) of the matrix
    elseif isa(x, Vector)
        return 1  # Return 1 if it's a column vector
    else
        type = typeof(x)
        error("Input data type: $type is neither a matrix or column vector")
    end
end

function remove_last_bias(model, Jacobian)
    """Removes last bias of model in jacobian, because of 'frozen parameter' """
    lastbias = length(Flux.params(model)[length(Flux.params(model))])
    Jacobian = Jacobian[:, 1:end-lastbias]
    return Jacobian
end

function model2lambda(model)
    """This functions converts a Flux Chain into in acceptible form for ReverseDiff"""
    _, restruct = Flux.destructure(model)

    # Lambda function
    m = (x, p) -> begin 
        mod = restruct(p)
        y = mod(x)
        return y
    end

    return m
end

function Jacobian_ReverseDiff(model, x, show_progress, diff_mode)
    N = check_dim(x)
    params,_ = Flux.destructure(model)

    l = length(model(x[:,1]))
    k = length(params)
    m = model2lambda(model)

    ### Tape
    if  diff_mode == 1
        tape = ReverseDiff.JacobianTape(m, (x[:,1],params))
        comp_tape = ReverseDiff.compile(tape)
    end

    ### Jacobian computations
    D = zeros(N*l, k)
    for i = 1:N
        if diff_mode == 1
            d = ReverseDiff.jacobian!(comp_tape, (x[:,i], params))[2]
        elseif diff_mode == 2
            d = ReverseDiff.jacobian(m, (x[:,i], params))[2]
        end

        D[(i-1)*l+1:i*l, :] .= d  
    end

    D = remove_last_bias(model, D)      

    return D
end

function Jacobian_Zygote(model,x, show_progress)
    if show_progress
        @time "\nZygote Jacobian" begin
            D = Zygote.jacobian(() -> model(x), Flux.params(model))
            D = hcat([(grad) for grad in D]...)
            D = remove_last_bias(model, D)         
        end
    else
        D = Zygote.jacobian(() -> model(x), Flux.params(model))
        D = hcat([(grad) for grad in D]...)
        D = remove_last_bias(model, D)        
    end

    return D
end

function Jacobian_Tracker(model,x, show_progress)
    ### REIKNA JACOBIAN
    l = length(model(x[:,1]))
    params, restruct = Flux.destructure(model)
    Jacobian = zeros(N*l,length(params))
    for i = 1:size(x)[2]
        h = (p) -> begin
            mod = restruct(p)
            y = mod(x[:,i])
        end
        d = Tracker.data(Tracker.jacobian(h, params))
        Jacobian[(i-1)*l+1:i*l, :] .= d
    end
    Jacobian = remove_last_bias(model, Jacobian)
    return Jacobian
end