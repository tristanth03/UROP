using Flux

struct DenseNTK
    weight  # Weight
    bias  # Bias
    σ  # activation function
end

function DenseNTK((in, out)::Pair, σ=identity; bias=true, init=Flux.randn32)
  weight = init(out, in)
  bias = Flux.create_bias(weight, bias, out)
  DenseNTK(weight, bias, σ)
end

function DenseNTK(in::Integer, out::Integer, σ=identity; bias=true, init=Flux.randn32)
  weight = init(out, in)
  bias = Flux.create_bias(weight, bias, out)
  DenseNTK(weight, bias, σ)
end

function (m::DenseNTK)(x::Vector)
    # Arbitrary code can go here, but note that everything will be differentiated.
    # Zygote does not allow some operations, like mutating arrays.
    σ = NNlib.fast_act(m.σ, x)  # replaces tanh => tanh_fast, etc
    return σ.((m.weight/sqrt(size(m.weight)[2]))*x .+ m.bias)
end

function (m::DenseNTK)(x::Array)
    # Arbitrary code can go here, but note that everything will be differentiated.
    # Zygote does not allow some operations, like mutating arrays.
    
    σ = NNlib.fast_act(m.σ, x)  # replaces tanh => tanh_fast, etc
    return σ.((m.weight/sqrt(size(m.weight)[2]))*x .+ m.bias)
end



Flux.@functor DenseNTK
