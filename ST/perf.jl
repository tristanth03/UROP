include("DenseNTK.jl");include("FastNTKMethods.jl");using Flux, LinearAlgebra, Random, Statistics, Zygote, ProgressMeter
h = 0.2; x0 = -1.0; xn = 1.0; activation = relu; dim = 1; N1 = 10_000
# We create some data (normalized)
x = hcat(range(x0,stop=xn,step=h)...)
f = x->sin(5*x)+cos(5*x)

model = Chain(DenseNTK(dim=>N1,activation),DenseNTK(N1=>dim))|>f64


Iterations = 100
L = []
y = map(f,x)
data = [(x,y)]
Loss(model,x,y) = Flux.mse(y,model(x))
params_t = []

for epoch = 1:Iterations

    Flux.train!(Loss,model,data,Descent(0.1))
    push!(L,Loss(model,x,y))
  
    p = deepcopy(Flux.params(model))
    push!(params_t,p)
    


end 
