using Flux,LinearAlgebra, Random, Statistics, Zygote, ProgressMeter
include("DenseNTK.jl")
include("FastNTKMethods.jl")

## model
## data
## kernel
## intensity (learning rate specific dynamics, t_step,epochs, param_num)



function LR_mapping(x,f,model,intensity,iterations)

    N = size(x)[2]
    N1 = size(Flux.params(model)[1])[1] 

    if iterations < 1*10^3
        println("More iterations are needed!")
    end
    if intensity == 1
        t_step = Int128(iterations*(1*10^-3))
        param_num1 = 1; param_num2 = Int128(round(N1*0.1)); layer = 1; gap=10;
    elseif intensity == 2
        t_step = Int128(iterations*(1*10^-3))
        param_num1 = 1; param_num2 = Int128(round(N1*0.5)); layer = 1; gap=2; 
    elseif intensity == 3
        t_step = Int128(iterations*(1*10^-3))
        param_num1 = 1; param_num2 = Int128(N1); layer = 1; gap=1;
    elseif intensity == 4
        t_step = Int128(iterations*(1*10^-2))
        param_num1 = 1; param_num2 = Int128(N1); layer = 1;  gap=1;# (Very heavy)
    end



    Nepoch = N*t_step

    y = map(f,x)
    data = [(x,y)]
    params_0 = deepcopy(Flux.params(model))
    K = kernel(model,x,false,3)
    eig = eigen(K).values

    
    Loss(model,x,y) = Flux.mse(y,model(x))

    lr = 1/eig[end]
    params_t = []
    for epoch = 1:Nepoch

        Flux.train!(Loss,model,data,Descent(lr))
        p = deepcopy(Flux.params(model))
        push!(params_t, p)

        # pt[epoch+1,:] = model[1].weight[pdt_num1:pdt_num2]
    end 
    LR = []

    J = Jacobian_Zygote(model,x, false)
    @showprogress for param_num = param_num1:gap:param_num2
        dfs = J[:,param_num]
        Z = zeros(N,1)


        for i = 1:N

            t = (i-1)*t_step
            model[1].weight .= params_0[1]
            model[1].bias .= params_0[2]
            model[2].weight .= params_0[3]
            model[2].bias .= params_0[4]
            if t == 0
                model[layer].weight[param_num] = params_0[layer][param_num]
            elseif t > 0
                model[layer].weight[param_num] = params_t[t][layer][param_num]
            end
            
            
            A = exp(-K*t)

            B = A*(model(x)-y)'
            C = (B'*dfs)
            Z[i,1] = C[1]

        end

        push!(LR,Z)
    end

    return LR,K,eig,N1
    
end


function LR_updt(h,x0,xn,detail,iterations)
    intensity = detail
    iterations = iterations
    LR_map,K,eig,N1 = LR_mapping(x,f,model,intensity,iterations)
    d = vcat(range(x0,stop=xn,step=h)...)
    MAT = [ones(size(d)[1],1) vcat(range(x0,stop=xn,step=h)...)]

    map_size = length(LR_map)
    LR_opting = zeros(map_size,1)
    for i = 1:map_size
        LR_opting[i,1] = (pinv(MAT)*LR_map[i])[2]
    end

    LR_opt2 = maximum(abs.(LR_opting))

    return LR_opt2*sqrt(N1),K,eig
end

