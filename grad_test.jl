include("NN_2_grad.jl")
include("models.jl")
include("packages.jl")

using Flux

X_training, Y_training, X_testing, Y_testing = load_MNIST()
model = model_3LS() # Simple model to play with

X_batch = X_training[:,1:100] # First 100 vectors
n = size(X_batch)[2]
m = size(X_batch)[2]
kernel_matrix  = zeros(Float32,n,m)


x1 = X_batch[:,1]
x2 = X_batch[:,2]

# # display(transpose(x1))

# # grads_x1_all = []
# # for i in 1:length(model(x1))
# #     grads_x1_i = gradient(params(model)) do
# #         y = model(x1)
# #         return y[i]
# #     end
# #     push!(grads_x1_all, grads_x1_i)
# # end


elapsed_time = @elapsed begin
    for i in 1:n
        x_i = X_batch[:, i]
        
        # Calculate gradients for x_i
        grads_xi_all = []
        for j in 1:length(model(x_i))
            grads_xi_j = gradient(Flux.params(model)) do
                y = model(x_i)
                return y[j]
            end
            push!(grads_xi_all, grads_xi_j)
        end
        
        for k in 1:m
            x_j = X_batch[:, k]
            
            # Calculate gradients for x_j
            grads_xj_all = []
            for l in 1:length(model(x_j))
                grads_xj_l = gradient(Flux.params(model)) do
                    y = model(x_j)
                    return y[l]
                end
                push!(grads_xj_all, grads_xj_l)
            end
            
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

# # # Calculate the kernel matrix entry for x1 and x2 using the entire gradients
# # kernel_entry = dot(grads_x1_all, grads_x1_all)

# # # Print the kernel matrix entry
# # println("Kernel matrix entry for x1 and x1:", kernel_entry)

# # Iterate over each pair of input vectors in X_batch
# # Initialize the kernel matrix








# model = model_3LS()  # Element-wise square

# # Calculate the diagonal of the Jacobian
# jacob = ForwardDiff.jacobian(model, x1)



