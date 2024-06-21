module ChainRulesKernelAbstractionsExt

import Adapt
import Atomix
import ChainRules
import GPUArrays
import KernelAbstractions as KA

using GPUArraysCore: AbstractGPUArray
using KernelAbstractions

function ChainRules.∇getindex!(dx::AbstractGPUArray, dy, inds...)
    kab = get_backend(dx)

    if KA.supports_atomics(kab)
        gids = GPUArrays.to_indices(dx, inds)
        idims = map(length, gids)
        Is = map(Adapt.adapt(GPUArrays.ToGPU(dy)), gids)
        scatter!(kab)(+, dx, dy, idims, Is...; ndrange=length(dy))
    else
        dx_cpu = Adapt.adapt(Array, dx)
        view(dx_cpu, Adapt.adapt(Array, inds)...) .+= Adapt.adapt(Array, dy)
        copyto!(dx, dx_cpu)
    end
    return dx
end

@kernel function scatter!(op, dest, src, idims, Is::Vararg{Any, N}) where N
    _scatter!(@index(Global), op, dest, src, idims, Is...)
end

@generated function _scatter!(i, op, dest, src, idims, Is::Vararg{Any, N}) where N
    quote
        is = @inbounds CartesianIndices(idims)[i]
        Base.Cartesian.@nexprs $N j -> I_j = @inbounds((Is[j])[is[j]])
        dv = src[i]
        Base.Cartesian.@ncall $N _accum! op dest dv j -> I_j
    end
end

function _accum!(op, dest, val, ids...)
    Atomix.modify!(Atomix.IndexableRef(dest, (ids...,)), op, val)
end

end
