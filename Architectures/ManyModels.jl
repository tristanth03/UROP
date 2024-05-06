# GAGNAÖFLUN
include("Architypes.jl")
using LinearAlgebra

"""
Returns a n-list of models (M) with the same architype

Works with ManyKernels() and ManyEigen()

This function is only to be used with defined depths and critical_widths.
Those vectors must have the same length

If provided a list of activations, that's one type of activation per model.
Those vectors must have the same length as depths and critical_widths
"""
function ManyModels(architype, dimIN, dimOUT, depths::Vector, activations, critical_widths::Vector,progress=false)
    if length(depths) != length(critical_widths)
        error("This function is only to be used with defined depths and critical_widths.\nThose vectors must have the same length")
    end
    if isa(activations, Vector)
        if length(depths) != length(activations)
            error("If provided a list of activations, that's one type of activation per model\nThose vectors must have the same length as depths and critical_widths")
        end
    elseif isa(activations, Function)
        activations = fill(activations, length(depths))
    else
        error("'activations' can only be a function or a vector of functions.")
    end

    M = [] # models

    if progress @showprogress desc="Creating $(length(depths)) models" for i in 1:length(depths)
            m = model_architype(architype, dimIN, dimOUT, depths[i], activations[i], critical_widths[i])
            push!(M, m)
        end
    else
        for i in 1:length(depths)
            m = model_architype(architype, dimIN, dimOUT, depths[i], activations[i], critical_widths[i])
            push!(M, m)
        end
    end

    return M
end

"""
Takes in ManyModels() object, (M) and returns n-list of NTK kernels (K)

Diffmodes:\n
1: ReverseDiff Tape\n
2: ReverseDiff No Tape\n
3: Zygote\n
4: Tracker
"""
function ManyKernels(x,M,diff_mode=4, progress=true)
    K = [] # Kernels

    if progress @showprogress desc="Creating $(length(M)) Kernels" for m in M
            k = kernel(m, x, false, diff_mode)
            push!(K, k)
        end
    else
        for m in M
            k = kernel(m, x, false, diff_mode)
            push!(K, k)
        end
    end

    return K
end

"""
Takes in ManyKernels() object, (K) and returns n-list of eigen-components (E)
"""
function ManyEigen(K,progress=true)
    E = [] # Eigen-components

    if progress @showprogress desc="Computing $(length(K)) Eigen-components" for k in K
            e = eigen(k)
            push!(E, e)
        end
    else
        for k in K
            e = eigen(k)
            push!(E, e)
        end
    end
end