using Flux
using Zygote
using MLDatasets
using LinearAlgebra

model = Chain(  Dense(2 => 2), Dense(2 => 1)) # W_2[1x2](W_1[2x2]x[2,1]+b_1[2x1])+b_2[1]

x1 = Float32[0.5852378, 0.62436277] # random datapoint
x2 = Float32[0.0976659, 0.55464536] # random datapoint

W1 = Flux.params(model)[1]  # W_1
b1 = Flux.params(model)[2]  # b_1

W1 .= ones(2,2)  #  Hér má setja eitthvað "fixed" fylki, breyti gildum í W1
b1 .= [1,1]

W2 = Flux.params(model)[3]  # W_1
b2 = Flux.params(model)[4]  # b_1

W2 .= ones(1,2)
b2 .= 1

# Handvirkt reiknaðar hlutaafleiður
gs_x1=Flux.gradient(() -> model(x1)[1],Flux.params(model))   # Reikna allar hlutaafleiður fyrir x1
gs_x2=Flux.gradient(() -> model(x2)[1],Flux.params(model))   # Reikna allar hlutaafleiður fyrir x2

grads_x1 = []
grads_x2 = []

for i = 1:length(Flux.params(model))
    push!(grads_x1, gs_x1[Flux.params(model)[i]])
    push!(grads_x2, gs_x2[Flux.params(model)[i]])
end

K1_1 = dot(grads_x1, grads_x1)
K1_2 = dot(grads_x1, grads_x2)

K2_1 = dot(grads_x2, grads_x1)
K2_2 = dot(grads_x2, grads_x2)

K = [K1_1 K1_2 ; K2_1 K2_2]

