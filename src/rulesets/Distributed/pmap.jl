#####
##### `pmap`
#####

# Now that there is a backwards rule for zip (albeit only in Zygote),
# it should be fine to deal with only a single collection X
function rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(pmap), f, p::AbstractWorkerPool, X; kwargs...)
    project_X = ProjectTo(X)

    darr = dfill([], (nworkers(p) + 1,), vcat(myid(), workers(p))) # Include own proc to handle empty worker pool

    function forw(x)
        y, back = rrule_via_ad(config, f, x)
        push!(darr[:L][1], back)
        return y, myid(), length(darr[:L][1])
    end

    ys_IDs_indices = pmap(forw, p, X; kwargs...)
    ys = getindex.(ys_IDs_indices, 1) # the primal values
    IDs = getindex.(ys_IDs_indices, 2) # remember which processors handled which elements of X
    indices = getindex.(ys_IDs_indices, 3) # remember the index of the pullback in the array on each processor
    output_sz = axes(ys)
 
    # create a list of positions in X handled by each processor
    unique_IDs = sort!(unique(IDs))
    T = eltype(eachindex(ys_IDs_indices))
    positions = [Vector{T}() for _ in 1:length(unique_IDs)]
    for i in eachindex(ys_IDs_indices)
        push!(positions[searchsortedfirst(unique_IDs, IDs[i])], i)
    end

    function pmap_pullback(Ȳ_raw)
        Ȳ = unthunk(Ȳ_raw)

        # runs the pullback for each position handled by proc ID in forward pass
        function run_backs(ID, positions)
            Ȳ_batch = Ȳ[positions]
            indices_batch = indices[positions]
            res_batch = remotecall_fetch(ID) do
                    asyncmap((ȳ, i) -> darr[:L][1][i](ȳ), Ȳ_batch, indices_batch) # run all the backs in a local asyncmap
                end 
            return res_batch
        end

        # combine the results from each proc into res = pmap((back, ȳ) -> back(ȳ), p, backs for each position, Ȳ)
        res_batches = asyncmap(run_backs, unique_IDs, positions)
        res = similar(res_batches[1], output_sz)

        for (positions, res_batch) in zip(positions, res_batches)
            res[positions] = res_batch
        end

        # extract f̄ and X̄ 
        f̄ = sum(first, res)
        X̄ = project_X(map(last, res))
        return (NoTangent(), f̄, NoTangent(), X̄)
    end

    return ys, pmap_pullback
end

