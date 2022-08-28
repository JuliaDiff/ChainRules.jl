#####
##### Comprehension: Iterators.map
#####

# Comprehension does guarantee iteration order. Thus its gradient must reverse.

function rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(collect), gen::G) where {G<:Base.Generator}
    @debug "collect generator"
    ys, backs = unzip_map(x -> rrule_via_ad(cfg, gen.f, x), gen.iter)
    proj_f = ProjectTo(gen.f)
    proj_iter = ProjectTo(gen.iter)
    function generator_pullback(dys_raw)
        dys = unthunk(dys_raw)
        dfs, dxs = unzip_map_reversed(|>, dys, backs)
        return (NoTangent(), Tangent{G}(; f = proj_f(sum(dfs)), iter = proj_iter(dxs)))
    end
    ys, generator_pullback
end

# Needed for Yota, but shouldn't these be automatic?
ChainRulesCore.rrule(::Type{<:Base.Generator}, f, iter) = Base.Generator(f, iter), dy -> (NoTangent(), dy.f, dy.iter)
ChainRulesCore.rrule(::Type{<:Iterators.ProductIterator}, iters) = Iterators.ProductIterator(iters), dy -> (NoTangent(), dy.iterators)

#=

          Yota.grad(xs -> sum(abs, [sin(x) for x in xs]), [1,2,3]pi/3)
Diffractor.gradient(xs -> sum(abs, [sin(x) for x in xs]), [1,2,3]pi/3)

          Yota.grad((xs, ys) -> sum(abs, [atan(x/y) for x in xs, y in ys]), [1,2,3]pi/3, [4,5])  # ERROR: all field arrays must have same shape
Diffractor.gradient((xs, ys) -> sum(abs, [atan(x/y) for x in xs, y in ys]), [1,2,3]pi/3, [4,5])  # ERROR: type Array has no field iterators

          Yota.grad(xs -> sum(abs, map(sin, xs)), [1,2,3]pi/3)
Diffractor.gradient(xs -> sum(abs, map(sin, xs)), [1,2,3]pi/3)  # fails internally

          Yota.grad(xs -> sum(abs, [sin(x/y) for (x,y) in zip(xs, 1:2)]), [1,2,3]pi/3)
Diffractor.gradient(xs -> sum(abs, [sin(x/y) for (x,y) in zip(xs, 1:2)]), [1,2,3]pi/3)

          Yota.grad(xs -> sum(abs, map((x,y) -> sin(x/y), xs, 1:2)), [1,2,3]pi/3)
Diffractor.gradient(xs -> sum(abs, map((x,y) -> sin(x/y), xs, 1:2)), [1,2,3]pi/3)


@btime Yota.grad($(rand(1000))) do xs
    sum(abs2, [sqrt(x) for x in xs])
end
# Yota         min 759.000 μs, mean 800.754 μs (22041 allocations, 549.62 KiB)
# Diffractor   min 559.000 μs, mean 622.464 μs (18051 allocations, 612.34 KiB)

# Zygote  min 3.198 μs, mean 6.849 μs (20 allocations, 40.11 KiB)


@btime Yota.grad($(rand(1000)), $(rand(1000))) do xs, ys
    zs = map(xs, ys) do x, y
        atan(x/y)
    end
    sum(abs2, zs)
end
# Yota + CR:      min 1.598 ms, mean 1.691 ms (38030 allocations, 978.75 KiB)
# Diffractor + CR:  min 767.250 μs, mean 847.640 μs (26045 allocations, 838.66 KiB)

# Zygote: min 13.417 μs, mean 22.896 μs (26 allocations, 79.59 KiB) -- 100x faster


=#


#####
##### `zip`
#####


function rrule(::typeof(zip), xs::AbstractArray...)
    function zip_pullback(dy)
        @debug "zip array pullback" summary(dy)
        dxs = _tangent_unzip(unthunk(dy))
        return (NoTangent(), map(_unmap_pad, xs, dxs)...)
    end
    function zip_pullback(dy::Tangent)
        @debug "zip Tangent pullback"
        return (NoTangent(), dy.is...)
    end
    zip_pullback(z::AbstractZero) = (NoTangent(), map(Returns(z), xs))
    return zip(xs...), zip_pullback
end

_tangent_unzip(xs::AbstractArray{Tangent{T,B}}) where {T<:Tuple, B<:Tuple} = unzip(reinterpret(B, xs))
_tangent_unzip(xs::AbstractArray) = unzip(xs)  # temp fix for Diffractor

# This is like unbroadcast, except for map's stopping-short behaviour, not broadcast's extension.
# Closing over `x` lets us re-use ∇getindex.
function _unmap_pad(x::AbstractArray, dx::AbstractArray)
    if length(x) == length(dx)
        ProjectTo(x)(reshape(dx, axes(x)))
    else
        @debug "_unmap_pad is extending gradient" length(x) == length(dx)
        i1 = firstindex(x)
        ∇getindex(x, vec(dx), i1:i1+length(dx)-1)
        # dx2 = vcat(vec(dx), similar(x, ZeroTangent, length(x) - length(dx)))
        # ProjectTo(x)(reshape(dx2, axes(x)))
    end 
end

# For testing
function rrule(::ComposedFunction{typeof(collect), typeof(zip)}, xs::AbstractArray...)
    y, back = rrule(zip, xs...)
    return collect(y), back
end

