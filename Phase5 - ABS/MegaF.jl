#--- A: packs and files
using Flux,Zygote,Random,CairoMakie,JuliaWorkspaces,LaTeXStrings,Statistics
include("DenseNTK.jl")
include("FastNTKMethods.jl") # we use the newest kernel
Random.seed!(123)

#--- B: input
# Model
dim = 1
N1 = 10_000
activation = relu
# Data
x0 = -1
xn = 1
h = 0.2
# kernel calculation type
kt = 3
# Parameter to "Freeze" at some time t_stop
layer = 1
param_type = "W"
t_stop = 2000
Nepoch = 5000
param_num1 = 100; param_num2 = 105

#--- C: model creation
# We create the usual model: 1 => 10_000 relu => 1
model = Chain(DenseNTK(dim=>N1,activation),DenseNTK(N1=>dim))|>f64

# We create some data (normalized)
x = hcat(range(x0,stop=xn,step=h)...)
f = x->sin(5*x)+cos(5*x)
y = map(f,x)

# First random fit with model (untrained)
ŷ = model(x)

# We define the loss function (MSE)
Loss(model,x,y) = Flux.mse(y,model(x))
loss = Loss(model,x,y)
data = [(x,y)]

#--- D: training
Kernel = kernel(model,x,true,kt) # Jacobian Zygote proves to keep accuracy as wanted
λ = eigen(Kernel).values

# We save parameters at t=0,1,...,Nepoch 
θ_0 = Flux.params(model)
θ_t = []

η = 1/λ[end]
@showprogress for epoch = 1:Nepoch
    Flux.train!(Loss,model,data,Descent(η))
    θ = deepcopy(Flux.params(model))
    push!(θ_t, θ)
end

#--- E: param initializing
# We create values for choosen parameter to look at [-1,1] with some steps
params = hcat(range(x0,stop=xn,step=h)...)

# initializing parameters
if t_stop == 0
    model[1].weight .= θ_0[1]
    model[1].bias .= θ_0[2]
    model[2].weight .= θ_0[3]
    model[2].bias .= θ_0[4]
elseif t_stop > 0
    model[1].weight .= θ_t[t_stop][1]
    model[1].bias .= θ_t[t_stop][2]
    model[2].weight .= θ_t[t_stop][3]
    model[2].bias .= θ_t[t_stop][4]
end

#--- F: the main loop 
MegaF = []
K = kernel(model,x,false,kt)
@showprogress for param_num = param_num1:param_num2

    if t_stop == 0
        model[1].weight .= θ_0[1]
        model[1].bias .= θ_0[2]
        model[2].weight .= θ_0[3]
        model[2].bias .= θ_0[4]
    elseif t_stop > 0
        model[1].weight .= θ_t[t_stop][1]
        model[1].bias .= θ_t[t_stop][2]
        model[2].weight .= θ_t[t_stop][3]
        model[2].bias .= θ_t[t_stop][4]
    end



    F_all = []
    N = length(x)
    if param_type == "W"
        for i = 1:N
            model[layer].weight[param_num] = params[i]
            
            local grads = []
            local dfs = []

            for j = 1:N
                local grad = Flux.gradient(()->model(x)[j],Flux.params(model))
                push!(grads,grad) 
            end
            
            for grad in grads
                local df = grad[model[layer].weight][param_num]
                push!(dfs,df)
            end
            
            # local K = kernel(model,x,false,kt)

            F = (exp(-K*t_stop)*(model(x)-y)')'*dfs
            push!(F_all,F)

        end
    elseif param_type == "B"
        for i = 1:N
            model[layer].bias[param_num] = params[i]
            
            local grads = []
            local dfs = []

            for j = 1:N
                local grad = Flux.gradient(()->model(x)[j],Flux.params(model))
                push!(grads,grad) 
            end
            
            for grad in grads
                local df = grad[model[layer].bias][param_num]
                push!(dfs,df)
            end
            
            

            F = (exp(-K*t_stop)*(model(x)-y)')'*dfs
            push!(F_all,F)

        end
    end

    push!(MegaF,F_all)
end

#--- G: plot
if param_type == "W"
    pt_label = "weight"
elseif param_type == "B"
    pt_label == "bias"
end

M = []
for i in MegaF
    push!(M,maximum(i)[1])
end
ymax = maximum(M)[1]

fig = Figure(fontsize=20)
ax = Axis(fig[1,1], title="Mega Function",
        xlabel=LaTeXString("\$\\theta_i\$"),
        ylabel=LaTeXString("\$\\mathscr{F}\$"),
        subtitle="Type:$pt_label | Layer:$layer",
        subtitlesize=16
        )

for z = 1:size(MegaF)[1]
    j = z + param_num1 - 1
    label = LaTeXString("\$\\theta_{$j}\$")
    CairoMakie.lines!(ax, Float64.(params)[:], map(x -> Float64(x[1]), MegaF[z])[:], label=label)
end


# axislegend(ax, position=:lt)

fig