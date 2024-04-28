using Zygote, LinearAlgebra, Random, CairoMakie
include("DenseNTK.jl")
include("FastNTKMethods.jl")
Random.seed!(123)

# --- Init of kernel --- #
kernel_mode = 3


function F(x, y, t, i, f0, θ_i)
    # F() = e^(-Kt)   * (ŷ-y) *   ∂f(x)/∂θi
    # F() =    A          B           C

    # --- Model setup
    params, restruct = Flux.destructure(f0)
    params[i] .= θi
    f = restruct(params)

    # A)
    global kernel_mode
    K = kernel(f, x, false, kernel_mode)
    λ, P = eigen(K)
    D = Diagonal(λ)

    A = P*exp(-D*t)*P^-1

    # B)
    B = (f(x)-y)

    # C)
    J = Jacobian_Zygote(f,x, false)
    C = J[:,i]

    return A*B*C
end