# AUTHOR: Axel Bjarkar Sigurjónsson
using Flux, LinearAlgebra, IterTools

function check_dim(x)
    """This function checks the appropriate  dimensions of input data"""
    if isa(x, Matrix)
        return size(x, 2)  # Returns the number of columns (width) of the matrix
    elseif isa(x, Vector)
        return 1  # Return 1 if it's a column vectoOkr
    else
        type = typeof(x)
        error("Input data type: $type is neither a matrix or column vector")
    end
end

function map_model(model, X)
    """This function evaluates model with repsect to appropriate dimensions"""
    N = check_dim(X)
    m = length(model(X[:,1]))  # Number of functions in the model output
    
    Ŷ = []

    if N == 1
        # If X is one datapoint
        if m == 1
            push!(Ŷ, model((X[:, 1]))[1])  # model outputs a 1-element array
        else
            push!(Ŷ, model((X[:, 1]))[:])  # model outputs a m-element array
        end
    else
        # If X is multiple datapoints
        if m == 1
            for i in 1:N
                push!(Ŷ, model(X[:, i])[1])  # model outputs a 1-element array
            end 
        else
            for i in 1:N
                push!(Ŷ, model(X[:, i])[:])  # model outputs a m-element array
            end 
        end
    end

    return Float64.(Ŷ)
end

function Df(model, x)
    # x: single datapoint
    m = length(model(x))
    k = sum(length, Flux.params(model)) # Total amount of params

    # Þetta anonymous function reiknar gradient fyrir hvert function í outputinu frá 1:m
    jac = (fi) -> Flux.jacobian(() -> model(x)[fi],Flux.params(model)) # anonymous function

    # Skilgreini tómt Jacobian fylki
    Jacob = zeros(k,m)

    for func_i = 1:m
        current_col = []
        for param_i = 1:length(Flux.params(model))
            push!(current_col, jac(func_i)[Flux.params(model)[param_i]]) # Fyrir hvern parametra W1, B1, W2...
        end
        current_col = collect(Iterators.flatten(current_col)) # Flatten, flet allt
        # --- Spurning hvort hægt sé að gera þetta skilvirkara?
        # --- Held samt að Iterators pakkinn eigi að vera nokkuð skilvirkur

        Jacob[:, func_i] .= current_col # geri current_col að næsta dálka vigri jacobian
    end

    return Jacob # Þetta er Df fylkið í bilblíunni
end

function kernel(model, x)
    N = check_dim(x)
    m = length(model(x[:,1]))  # Number of functions in the model output
    K = zeros(N*m, N*m)
    
    for i = 1:N
        for j = 1:N
            block = Df(model, x[:,i])' * Df(model, x[:,j])
            K[(i-1)*m+1:i*m, (j-1)*m+1:j*m] .= block
        end
    end

    return K
end