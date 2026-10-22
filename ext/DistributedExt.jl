module DistributedExt

using ChainRulesCore
using Distributed

@non_differentiable Distributed.CachingPool(::Any...)
@non_differentiable Distributed.WorkerPool(::Any...)
@non_differentiable workers()

end # module
