#####
##### broadcast
#####

"""
    unzip_broadcast(f, args...)

For a function `f` which returns a tuple, this is `== unzip(broadcast(f, args...))`,
but performed using `StructArrays` for efficiency. Used in the gradient of broadcasting.

# Examples
```
julia> using ChainRules: unzip_broadcast, unzip

julia> unzip_broadcast(x -> (x,2x), 1:3)
([1, 2, 3], [2, 4, 6])

julia> mats = @btime unzip_broadcast((x,y) -> (x+y, x-y), 1:1000, transpose(1:1000));  # 2 arrays, each 7.63 MiB
  min 1.776 ms, mean 20.421 ms (4 allocations, 15.26 MiB)

julia> mats == @btime unzip(broadcast((x,y) -> (x+y, x-y), 1:1000, transpose(1:1000)))  # intermediate matrix of tuples
  min 2.660 ms, mean 40.007 ms (6 allocations, 30.52 MiB)
true
```
"""
function unzip_broadcast(f::F, args...) where {F}
    T = Broadcast.combine_eltypes(f, args)
    if isconcretetype(T)
        T <: Tuple || throw(ArgumentError("""unzip_broadcast(f, args) only works on functions returning a tuple,
            but f = $(sprint(show, f)) returns type T = $T"""))
    end
    bc = Broadcast.instantiate(Broadcast.broadcasted(f, args...))
    bcs = Broadcast.BroadcastStyle(typeof(bc))
    if bcs isa AbstractGPUArrayStyle
        # This is a crude way to allow GPU arrays, not currently tested, TODO.
        # See also https://github.com/JuliaArrays/StructArrays.jl/issues/150
        return unzip(broadcast(f, args...))
    elseif bcs isa Broadcast.AbstractArrayStyle
        return StructArrays.components(StructArray(bc))
    else
        return unzip(broadcast(f, args...))  # e.g. tuples
    end
    # TODO maybe this if-else can be replaced by methods of `unzip(:::Broadcast.Broadcasted)`?
end

function ChainRulesCore.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(unzip_broadcast), f::F, args...) where {F}
    y, back = rrule_via_ad(cfg, broadcast, f, args...)
    z = unzip(y)
    function untuplecast(dz)
        # dy = StructArray(map(unthunk, dz))  # fails for e.g. StructArray(([1,2,3], ZeroTangent()))
        dy = broadcast(tuple, map(unthunk, dz)...)
        db, df, dargs... = back(dy)
        return (db, sum(df), map(unbroadcast, args, dargs)...)
    end
    untuplecast(dz::AbstractZero) = (NoTangent(), NoTangent(), map(Returns(dz), args))
    return z, untuplecast
end

# This is for testing, but the tests using it don't work.
function rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(collect∘unzip_broadcast), f, args...)
    y, back = rrule(cfg, unzip_broadcast, f, args...)
    return collect(y), back
end

#####
##### map
#####

"""
    unzip_map(f, args...)

For a function `f` which returns a tuple, this is `== unzip(map(f, args...))`,
but performed using `StructArrays` for efficiency.
"""
function unzip_map(f::F, args...) where {F}
    T = Broadcast.combine_eltypes(f, args)
    if isconcretetype(T)
        T <: Tuple || throw(ArgumentError("""unzip_map(f, args) only works on functions returning a tuple,
        but f = $(sprint(show, f)) returns type T = $T"""))
    end
    return StructArrays.components(StructArray(Iterators.map(f, args...)))
end

unzip_map(f::F, args::Tuple...) where {F} = unzip(map(f, args...))
# unzip_map(f::F, args::NamedTuple...) where {F} = unzip(map(f, args...))

unzip_map(f::F, args::AbstractGPUArray...) where {F} = unzip(map(f, args...))

"""
    unzip_map_reversed(f, args...)

For a pure function `f` which returns a tuple, this is `== unzip(map(f, args...))`.
But the order of evaluation is should be the reverse.
Does NOT handle `zip`-like behaviour.
"""
function unzip_map_reversed(f::F, args...) where {F}
    T = Broadcast.combine_eltypes(f, args)
    if isconcretetype(T)
        T <: Tuple || throw(ArgumentError("""unzip_map_reversed(f, args) only works on functions returning a tuple,
        but f = $(sprint(show, f)) returns type T = $T"""))
    end
    len1 = length(first(args))
    all(a -> length(a)==len1, args) || error("unzip_map_reversed does not handle zip-like behaviour.")
    return map(reverse!!, unzip_map(f, map(_safereverse, args)...))
end

# This avoids MethodError: no method matching iterate(::Base.Iterators.Reverse{Tangent{Tuple{Float64, Float64}, Tuple{Float64, Float64}}}) on 1.6
_safereverse(x) = VERSION > v"1.7" ? Iterators.reverse(x) : reverse(x)

function unzip_map_reversed(f::F, args::Tuple...) where {F}
    len1 = length(first(args))
    all(a -> length(a)==len1, args) || error("unzip_map_reversed does not handle zip-like behaviour.")
    return map(reverse, unzip(map(f, map(reverse, args)...)))
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
reverse!!(x::AbstractArray{<:AbstractZero}) = x
reverse!!(x) = reverse(x)

frule((_, xdot), ::typeof(reverse!!), x) = reverse!!(x), reverse!!(xdot)

function rrule(::typeof(reverse!!), x)
    reverse!!_back(dy) = (NoTangent(), reverse(unthunk(dy)))
    return reverse!!(x), reverse!!_back
end


#####
##### unzip
#####

"""
    unzip(A)

Converts an array of tuples into a tuple of arrays.
Eager. Will work by `reinterpret` when possible.

```jldoctest
julia> ChainRules.unzip([(1,2), (30,40), (500,600)])  # makes two new Arrays:
([1, 30, 500], [2, 40, 600])

julia> typeof(ans)
Tuple{Vector{Int64}, Vector{Int64}}

julia> ChainRules.unzip([(1,nothing) (3,nothing) (5,nothing)])  # this can reinterpret:
([1 3 5], [nothing nothing nothing])

julia> ans[1]
1×3 reinterpret(Int64, ::Matrix{Tuple{Int64, Nothing}}):
 1  3  5
```
"""
function unzip(xs::AbstractArray)
    x1 = first(xs)
    x1 isa Tuple || throw(ArgumentError("unzip only accepts arrays of tuples"))
    N = length(x1)
    return unzip(xs, Val(N))  # like Zygote's unzip. Here this is the fallback case.
end

@generated function unzip(xs, ::Val{N}) where {N}
    each = [:(map($(Get(i)), xs)) for i in 1:N]
    Expr(:tuple, each...)
end

function unzip(xs::AbstractArray{Tuple{T}}) where {T}
    if isbitstype(T)
        (reinterpret(T, xs),)  # best case, no copy
    else
        (map(only, xs),)
    end
end

@generated function unzip(xs::AbstractArray{Ts}) where {Ts<:Tuple}
    each = if count(!Base.issingletontype, Ts.parameters) < 2 && all(isbitstype, Ts.parameters)
        # good case, no copy of data, some trivial arrays
        [Base.issingletontype(T) ? :(similar(xs, $T)) : :(reinterpret($T, xs)) for T in Ts.parameters]
    else
        [:(map($(Get(i)), xs)) for i in 1:length(fieldnames(Ts))]
    end
    Expr(:tuple, each...)
end

"""
    unzip(t)

Also works on a tuple of tuples:

```jldoctest
julia> unzip(((1,2), (30,40), (500,600)))
((1, 30, 500), (2, 40, 600))
```
"""
function unzip(xs::Tuple)
    x1 = first(xs)
    x1 isa Tuple || throw(ArgumentError("unzip only accepts arrays or tuples of tuples"))
    return ntuple(i -> map(Get(i), xs),length(x1))
end

struct Get{i} end
Get(i) = Get{Int(i)}()
(::Get{i})(x) where {i} = x[i]

function ChainRulesCore.rrule(::typeof(unzip), xs::AbstractArray{T}) where {T <: Tuple}
    function rezip(dy)
        dxs = broadcast(xs, unthunk.(dy)...) do x, ys...
            ProjectTo(x)(Tangent{T}(ys...))
        end
        return (NoTangent(), dxs)
    end
    rezip(dz::AbstractZero) = (NoTangent(), dz)
    return unzip(xs), rezip
end

function ChainRulesCore.rrule(::typeof(unzip), xs::Tuple)
    function rezip_2(dy)
        dxs = broadcast(xs, unthunk.(dy)...) do x, ys...
            Tangent{typeof(x)}(ys...)
        end
        return (NoTangent(), ProjectTo(xs)(dxs))
    end
    rezip_2(dz::AbstractZero) = (NoTangent(), dz)
    return unzip(xs), rezip_2
end
