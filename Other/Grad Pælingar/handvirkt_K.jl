# Þetta forrit er til að staðfesta K sem var reiknað á blaði, sjá PDF í þessu folderi
using Flux
using Zygote
using MLDatasets
using LinearAlgebra

model = Chain(  Dense(1 => 2), Dense(2 => 1)) # W_2[1x2](W_1[2x1]x[1]+b_1[2x1])+b_2[1]

x1 = [1.0]
x2 = [2.0]

# Skilgreini parametra
W1 = Flux.params(model)[1]  # W_1
b1 = Flux.params(model)[2]  # b_1
W2 = Flux.params(model)[3]  # W_1
b2 = Flux.params(model)[4]  # b_1

# Breyti gildum í parametrum
W1 .= ones(2,1)  #  Hér má setja eitthvað "fixed" fylki, breyti gildum í W1
b1 .= [0;1]

W2 .= [1 0]
b2 .= 0

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

K = [K1_1 K1_2 ; K2_1 K2_2];