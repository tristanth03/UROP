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

function Fkernel(model, x, show_progress=false)
    N = check_dim(x)                    # Number of datapoints
    m = length(model(x[:,1]))           # Number of functions in the model output

    Θ = zeros(N*(m*m), N*(m*m))         # Kernel is depecicted in research papers and Wikipedia

    D = Flux.jacobian(() -> model(x),Flux.params(model))
    D = hcat([grad for grad in D]...)
    D = D[:,1:end-1]                    # To skip the last bias

    ∂(f,x) = D[(f-m)+(x*m),:]           # Used in nested loop for readability

    if show_progress
        progress_Θ = Progress(m, 1, "Computing Θ:", 50)
    end

    for k = 1:m
        for l = 1:m
            mini_kernel = zeros(N,N)
            for i = 1:N
                for j = 1:N
                    mini_kernel[i,j] = dot(∂(k,i),∂(l,j))
                end
            end
            # Add mini_kernel to the corresponding portion of Θ
            Θ[(k-1)*N+1:k*N, (l-1)*N+1:l*N] .= mini_kernel
        end
        if show_progress
            next!(progress_Θ)           # Increment progress meter Θ
        end
    end

    return Θ
end