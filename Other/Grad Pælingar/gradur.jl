using Flux
using Zygote
using MLDatasets
using LinearAlgebra

model = Chain(  Dense(2 => 2), Dense(2 => 1)) # W_2[1x2](W_1[2x2]x[2,1]+b_1[2x1])+b_2[1]

x = Float32[0.5852378, 0.62436277] # random datapoint

W1 = Flux.params(model)[1]  # W_1
b1 = Flux.params(model)[2]  # b_1

W1 .= ones(2,2)  #  Hér má setja eitthvað "fixed" fylki, breyti gildum í W1
b1 .= [1,1]

W2 = Flux.params(model)[3]  # W_1
b2 = Flux.params(model)[4]  # b_1

W2 .= ones(1,2)
b2 .= 1

y=model(x)


gs=Flux.gradient(() -> model(x)[1],Flux.params(model))   # Reikna allar hlutaafleiður

# Filter out the undesired entry associated with :(Main.x)
filter!(x -> first(x) != :(Main.x), gs.grads)

# Collect the remaining gradient values
all_values = values(gs.grads)

grads_nums = []

for matrix in all_values
    push!(grads_nums, matrix...)
end

K1_1 = dot(grads_nums, grads_nums)

println(K1_1)

