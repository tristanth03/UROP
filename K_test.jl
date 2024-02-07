include("NN.jl")

model = Chain(  Dense(784 => 2), Dense(2 => 1)) # W_2[1x2](W_1[2x1]x[1]+b_1[2x1])+b_2[1]

x1 = load_MNIST()[1][:,1]
x2 = load_MNIST()[1][:,2]

# reikna gradient fyrir gefið x
gs = x -> Flux.gradient(() -> model(x)[1],Flux.params(model)) # anonymous function

grads_x1 = []
grads_x2 = []

# Ná í grads gildi
for i = 1:length(Flux.params(model))
    push!(grads_x1, gs(x1)[Flux.params(model)[i]][:])
    push!(grads_x2, gs(x2)[Flux.params(model)[i]][:])
end

# fletja
grads_x1 = reduce(vcat, grads_x1)
grads_x2 = reduce(vcat, grads_x2)

K1_1 = dot(grads_x1, grads_x1)
K1_2 = dot(grads_x1, grads_x2)

K2_1 = dot(grads_x2, grads_x1)
K2_2 = dot(grads_x2, grads_x2)

oldK = [K1_1 K1_2 ; K2_1 K2_2];

newK = kernel(model,2)