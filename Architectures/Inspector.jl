using JLD2

function meta_data(title, architype, nθ, depth, widths, activations)
    return [title, architype, nθ, depth, widths, activations]
end


function zip(meta_data, M,K,E)
    data = [meta_data, M, K, E]
    return data
end

function save_to_jld2(folder, filename, datapoints, object_name)
    filename = joinpath(folder, filename*".jld2")
    save(filename,object_name,datapoints)
end
