tup2(x) = Tuple{Any,Any}(x)  # temp fix for Diffractor

#####
##### Comprehension: Iterators.map
#####

# Comprehension does guarantee iteration order. Thus its gradient must reverse.

function rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(collect), gen::G) where {G<:Base.Generator}
    # ys, backs = unzip_map(x -> rrule_via_ad(cfg, gen.f, x)|>tup2, gen.iter)
    ys, backs = unzip(map(x -> rrule_via_ad(cfg, gen.f, x)|>tup2, gen.iter))
    proj_f = ProjectTo(gen.f)
    proj_iter = ProjectTo(gen.iter)
    function generator_pullback(dys_raw)
        dys = unthunk(dys_raw)
        # dfs, dxs = unzip_map(|>, Iterators.reverse(dys), Iterators.reverse(backs))
        dfs, dxs = unzip(map(|>, Iterators.reverse(dys), Iterators.reverse(backs)))
        return (NoTangent(), Tangent{G}(; f = proj_f(sum(dfs)), iter = proj_iter(reverse!!(dxs))))
    end
    ys, generator_pullback
end

"""
    reverse!!(x)

Reverses `x` in-place if possible, according to `ChainRulesCore.is_inplaceable_destination`.
Only safe if you are quite sure nothing else closes over `x`.
"""
function reverse!!(x::AbstractArray)
    if ChainRulesCore.is_inplaceable_destination(x)
        Base.reverse!(x)
    else
        Base.reverse(x)
    end
end
frule((_, xdot), ::typeof(reverse!!), x::AbstractArray) = reverse!!(x), reverse!!(xdot)
function rrule(::typeof(reverse!!), x::AbstractArray)
    reverse!!_back(dy) = (NoTangent(), reverse(unthunk(dy)))
    return reverse!!(x), reverse!!_back
end

# Needed for Yota, but shouldn't these be automatic?
ChainRulesCore.rrule(::Type{<:Base.Generator}, f, iter) = Base.Generator(f, iter), dy -> (NoTangent(), dy.f, dy.iter)
ChainRulesCore.rrule(::Type{<:Iterators.ProductIterator}, iters) = Iterators.ProductIterator(iters), dy -> (NoTangent(), dy.iterators)

#=

          Yota.grad(xs -> sum(abs, [sin(x) for x in xs]), [1,2,3]pi/3)
Diffractor.gradient(xs -> sum(abs, [sin(x) for x in xs]), [1,2,3]pi/3)

          Yota.grad((xs, ys) -> sum(abs, [atan(x/y) for x in xs, y in ys]), [1,2,3]pi/3, [4,5])
Diffractor.gradient((xs, ys) -> sum(abs, [atan(x/y) for x in xs, y in ys]), [1,2,3]pi/3, [4,5])

          Yota.grad(xs -> sum(abs, map(sin, xs)), [1,2,3]pi/3)
Diffractor.gradient(xs -> sum(abs, map(sin, xs)), [1,2,3]pi/3)  # fails internally

          Yota.grad(xs -> sum(abs, [sin(x/y) for (x,y) in zip(xs, 1:2)]), [1,2,3]pi/3)
Diffractor.gradient(xs -> sum(abs, [sin(x/y) for (x,y) in zip(xs, 1:2)]), [1,2,3]pi/3)

          Yota.grad(xs -> sum(abs, map((x,y) -> sin(x/y), xs, 1:2)), [1,2,3]pi/3)
Diffractor.gradient(xs -> sum(abs, map((x,y) -> sin(x/y), xs, 1:2)), [1,2,3]pi/3)


@btime Yota.grad($(rand(1000))) do xs
    sum(abs2, [sqrt(x) for x in xs])
end
# Yota             min 1.134 ms, mean 1.207 ms (22017 allocations, 548.50 KiB)
# Diffractor         min 936.708 μs, mean 1.020 ms (18028 allocations, 611.25 KiB)
# without unzip_map  min 734.292 μs, mean 810.341 μs (13063 allocations, 517.97 KiB)

# Zygote  min 6.117 μs, mean 11.287 μs (24 allocations, 40.31 KiB)


@btime Yota.grad($(rand(1000)), $(rand(1000))) do xs, ys
    zs = map(xs, ys) do x, y
        atan(x/y)
    end
    sum(abs2, zs)
end
# Yota + CR:        min 2.643 ms, mean 2.781 ms (35011 allocations, 915.19 KiB)
# Diffractor + CR:  min 1.184 ms, mean 1.285 ms (23026 allocations, 775.09 KiB)
# without unzip_map   min 947.084 μs, mean 1.036 ms (18062 allocations, 697.86 KiB)

# Zygote: min 21.291 μs, mean 36.456 μs (26 allocations, 79.59 KiB)


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
_tangent_unzip(xs::AbstractArray) = unzip(xs)  # Diffractor

function _unmap_pad(x::AbstractArray, dx::AbstractArray)
    if length(x) == length(dx)
        ProjectTo(x)(reshape(dx, axes(x)))
    else
        i1 = firstindex(x)
        ∇getindex(x, vec(dx), i1:i1+length(dx)-1)
        # dx2 = vcat(vec(dx), similar(x, ZeroTangent, length(x) - length(dx)))
        # ProjectTo(x)(reshape(dx2, axes(x)))
    end 
end



