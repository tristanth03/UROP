elapsed_time = @elapsed begin
include("NN_2_grad.jl")
include("models.jl")
include("packages.jl")

using Flux, Zygote
using LinearAlgebra

X_training, Y_training, X_testing, Y_testing = load_MNIST()
model = model_5LS2048()  # Simple model to play with

X_batch = X_training[:, 1:10]  # First 100 vectors
n = size(X_batch, 2)
m = size(X_batch, 2)
kernel_matrix = zeros(Float32, n, m)

jacobians = []
for i in 1:n
    x_i = X_batch[:, i]
    @show x_i
    push!(jacobians, jacobian(Flux.params(model)) do
        y = model(x_i)
        return y  # Remove the sum(y) to keep the vector y
    end)
end

@show jacobians

for i in 1:n
    grads_xi_all = jacobians[i]
    
    for k in 1:m
        x_j = X_batch[:, k]
        grads_xj_all = jacobians[k]
        
        # Calculate the kernel matrix entry for x_i and x_j
        kernel_entry = dot(grads_xi_all, grads_xj_all)
        
        # Store the entry in the kernel matrix
        kernel_matrix[i, k] = kernel_entry
    end
end
end

# Print the resulting kernel matrix
println("Kernel Matrix:")
display(kernel_matrix)
println("Elapsed time: $elapsed_time seconds")