include("NN_2_grad.jl")
include("models.jl")
include("packages.jl")

X_training, Y_training, X_testing, Y_testing = load_MNIST()
model = model_3LS() # Simple model to play with

X_batch = X_training[:,1:60] # First 60 vectors
n = size(X_batch)[2]
m = size(X_batch)[2]
kernel_matrix  = zeros(Float32,n,m)

# Create random input vectors x1 and x2
x1 = X_batch[:,1]
x2 = X_batch[:,2]

# display(transpose(x1))

grads_x1_all = []
for i in 1:length(model(x1))
    grads_x1_i = gradient(params(model)) do
        y = model(x1)
        return y[i]
    end
    push!(grads_x1_all, grads_x1_i)
end




# Calculate the kernel matrix entry for x1 and x2 using the entire gradients
kernel_entry = dot(grads_x1_all, grads_x1_all)

# Print the kernel matrix entry
println("Kernel matrix entry for x1 and x1:", kernel_entry)

