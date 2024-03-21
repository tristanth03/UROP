using Flux, LinearAlgebra, ProgressMeter

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

function kernel(model, x, show_progress=false)
    N = check_dim(x)                    # Number of datapoints
    m = length(model(x[:,1]))           # Number of functions in the model output

    Θ = zeros(N*m, N*m)                 # Kernel as depecicted in research papers and Wikipedia

    D = Flux.jacobian(() -> model(x),Flux.params(model))
    D = hcat([grad for grad in D]...)
    D = D[:,1:end-1]                    # To skip the last bias

    ∂(f,x) = D[(f-m)+(x*m),:]           # Used in nested loop for readability

    if show_progress
        progress_Θ = Progress(N * m*m, 1)   # Initialize progress meter for Θ
    end

    for k = 1:m
        for l = 1:m
            mini_kernel = zeros(N,N)
            for i = 1:N
                for j = 1:N
                    mini_kernel[i,j] = dot(∂(k,i),∂(l,j))
                end
                if show_progress
                    next!(progress_Θ)   # Increment progress meter Θ
                end
            end

            # Add mini_kernel to the corresponding portion of Θ
            Θ[(k-1)*N+1:k*N, (l-1)*N+1:l*N] .= mini_kernel
        end
    end

    if show_progress
        finish!(progress_Θ)  # Finish progress meter
    end

    return Θ
end

function faster_kernel(model, x, show_progress=false)
    N = check_dim(x)                    # Number of datapoints
    m = length(model(x[:,1]))           # Number of functions in the model output

    Θ = zeros(N*m, N*m)                 # Kernel as depecicted in research papers and Wikipedia

    D = Flux.jacobian(() -> model(x), Flux.params(model))
    D = hcat([grad for grad in D]...)
    D = D[:,1:end-1]                    # To skip the last bias

    ∂x(i) = (((i-1)*m)+1:i*m)           # Used in nested loop for readability

    if show_progress
        prog = Progress(N^2, 1, "Computing Kernel...", 50)
    end

    for i = 1:N
        for j = 1:N
            block = D[∂x(i),:]*D[∂x(j),:]'        # Will produce mxm matrix
            Θ[∂x(i),∂x(j)] .= block
            
            if show_progress
                next!(prog)
            end
        end
    end

    return Θ
end

function fastest_kernel(model, x, show_progress=false)
    N = check_dim(x)                    # Number of datapoints
    m = length(model(x[:,1]))           # Number of functions in the model output

    Θ = zeros(Float32, N*m, N*m)        # Kernel as depicted in research papers and Wikipedia

    D = Flux.jacobian(() -> model(x), Flux.params(model))
    D = hcat([Float32.(grad) for grad in D]...)
    D = D[:,1:end-1]                    # To skip the last bias

    ∂x(i) = (((i-1)*m)+1:i*m)           # Used in nested loop for readability

    if show_progress
        prog = Progress(N^2, 1, "Computing Kernel...", 50)
    end

    for i = 1:N
        for j = 1:N
            block = D[∂x(i),:]*D[∂x(j),:]'        # Will produce mxm matrix
            Θ[∂x(i),∂x(j)] .= block
            
            if show_progress
                next!(prog)
            end
        end
    end

    return Θ
end

function ult_kernel(model, x)
    t1 = time()  # Record the start time

    D = Flux.jacobian(() -> model(x), Flux.params(model))
    D = hcat([Float32.(grad) for grad in D]...)
    D = D[:, 1:end-1]  # To skip the last bias

    t2 = time()  # Record the time after calculating D

    Θ = D * D'

    t3 = time()  # Record the time after calculating Θ

    println("Time taken to calculate D: ", t2 - t1)
    println("Time taken to calculate Θ: ", t3 - t2)

    return Θ
end
