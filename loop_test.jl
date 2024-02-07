using Flux
using LinearAlgebra
using IterTools
using Zygote
using MLDatasets

include("NN.jl")

model = Chain(  Dense(1 => 2), Dense(2 => 1)) # W_2[1x2](W_1[2x1]x[1]+b_1[2x1])+b_2[1]

x = [[1.0], [2.0]]
n = length(x)

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
 gs = xi-> Flux.gradient(() -> model(xi)[1],Flux.params(model)) # anonymous function
    
 all_grads = []
 
 # Collect numerical values
 for i in 1:n
     current_grad = []
     for j in 1:length(Flux.params(model))
         push!(current_grad, gs(x[i])[Flux.params(model)[j]]) # [:] til að fletja
     end

     @show current_grad

     current_grad = collect(Iterators.flatten(current_grad)) # flatten
    
     @show current_grad

     push!(all_grads, current_grad)
     println("\n")
 end
