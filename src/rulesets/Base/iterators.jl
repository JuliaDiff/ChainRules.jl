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
    end 
end

# For testing
function rrule(::ComposedFunction{typeof(collect), typeof(zip)}, xs::AbstractArray...)
    y, back = rrule(zip, xs...)
    return collect(y), back
end

