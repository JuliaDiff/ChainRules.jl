using ChainRules
using GPUArrays
using Zygote
using AMDGPU
using KernelAbstractions
using KernelAbstractions: @atomic

function _accum!(dest, val, ids...)
    # TODO support passing `op`
    @atomic dest[ids...] += val
end

@generated function _scatter!(i, dest, src, idims, Is::Vararg{Any, N}) where N
    quote
        is = @inbounds CartesianIndices(idims)[i]
        Base.Cartesian.@nexprs $N j -> I_j = @inbounds((Is[j])[is[j]])
        dv = dest[i]
        Base.Cartesian.@ncall $N _accum! src dv j -> I_j
    end
end

@kernel function scatter!(dest, src, idims, Is::Vararg{Any, N}) where N
    _scatter!(@index(Global), dest, src, idims, Is...)
end

function main()
    x = ROCArray(zeros(Float32, 16, 4, 2, 3))
    y = ROCArray(ones(Float32, 6, 2, 2))
    ids = ([4, 1, 4, 3, 2, 1], 1, :, 3)

    gids = GPUArrays.to_indices(x, ids)
    idims = map(length, gids)
    Is = map(AMDGPU.Adapt.adapt(GPUArrays.ToGPU(y)), gids)

    kab = get_backend(x)
    scatter!(kab, 256)(y, x, idims, Is...; ndrange=length(y))
    @show y
    @show Array(x)[:, 1, 1, 3]

    # @show x[ids...]
    # x[ids...] .+= y
    # return

    # Δ = ROCArray(ones(Float32, 1))

    # y, back = Zygote.pullback(x) do x
    #     # xd = x[[4, 3, 2, 1], :, 1, [3, 1]]
    #     xd = x[]
    #     sum(xd; dims=(1:ndims(xd)...,))
    # end
    # println("===============")
    # back(Δ)
    return
end
main()
