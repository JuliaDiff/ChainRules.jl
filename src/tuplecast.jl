
"""
    tuplecast(f, args...)

For a function `f` which returns a tuple, this is `== unzip(broadcast(f, args...))`,
but performed using `StructArrays` for efficiency.

# Examples
```
julia> using ChainRules: tuplecast, unzip

julia> tuplecast(x -> (x,2x), 1:3)
([1, 2, 3], [2, 4, 6])

julia> mats = @btime tuplecast((x,y) -> (x+y, x-y), 1:1000, transpose(1:1000));  # 2 arrays, each 7.63 MiB
  min 1.776 ms, mean 20.421 ms (4 allocations, 15.26 MiB)

julia> mats == @btime unzip(broadcast((x,y) -> (x+y, x-y), 1:1000, transpose(1:1000)))  # intermediate matrix of tuples
  min 2.660 ms, mean 40.007 ms (6 allocations, 30.52 MiB)
true
```
"""
function tuplecast(f::F, args...) where {F}
    T = Broadcast.combine_eltypes(f, args)
    if isconcretetype(T)
        T <: Tuple || throw(ArgumentError("""tuplecast(f, args) only works on functions returning a tuple,
            but f = $(sprint(show, f)) returns type T = $T"""))
    end
    # if any(a -> a isa CuArray, args)
    #     return unzip(broadcast(f, args...))
    # end
    bc = Broadcast.instantiate(Broadcast.broadcasted(f, args...))
    StructArrays.components(StructArray(bc))
end

function ChainRulesCore.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(tuplecast), f::F, args...) where {F}
    y, back = rrule_via_ad(cfg, broadcasted, f, args...)
    z = unzip(y)
    function untuplecast(dz)
        dy = StructArray(map(unthunk, dz))
        db, df, dargs... = back(dy)
        (db, sum(df), map(unbroadcast, args, dargs)...)
    end
    return z, untuplecast
end

# function rrule(cfg::RCR, ::typeof(collect∘tuplecast), f, args...)
#     y, back = rrule(cfg, tuplecast, f, args...)
#     return collect(y), back
# end

"""
    tuplemap(f, args...)

For a function `f` which returns a tuple, this is `== unzip(map(f, args...))`,
but performed using `StructArrays` for efficiency.
"""
function tuplemap(f::F, args...) where {F}
    T = Broadcast.combine_eltypes(f, args)
    if isconcretetype(T)
        T <: Tuple || throw(ArgumentError("""tuplemap(f, args) only works on functions returning a tuple,
            but f = $(sprint(show, f)) returns type T = $T"""))
    end
    # if any(a -> a isa CuArray, args)
    #     return unzip(map(f, args...))
    # end
    StructArrays.components(StructArray(Iterators.map(f, args...)))
end

# function ChainRulesCore.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(tuplemap), f::F, args...) where {F}
#     y, back = rrule(cfg, map, f, xs...)  # won't work, but also, you want the lazier fwd
#     z = unzip(y)
#     function untuplemap(dz)
#         dy = StructArray(map(unthunk, dz))
#         back(dy)
#     end
#     return unzip(xs), untuplemap
# end

"""
    unzip(A)

Converts an array of tuples into a tuple of arrays.
Eager. Will work by `reinterpret` when possible.

```jldoctest
julia> ChainRules.unzip([(1,2), (3,4), (5,6)])  # makes two new Arrays:
([1, 3, 5], [2, 4, 6])

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
    unzip(xs, Val(N))  # like Zygote's unzip, here this is the fallback case.
end

@generated function unzip(xs, ::Val{N}) where {N}
    each = [:(map($(Get(i)), xs)) for i in 1:N]
    Expr(:tuple, each...)
end

unzip(xs::AbstractArray{Tuple{T}}) where {T} = (reinterpret(T, xs),)  # best case, no copy

@generated function unzip(xs::AbstractArray{Ts}) where {Ts<:Tuple}
    each = if count(!Base.issingletontype, Ts.parameters) < 2
        # good case, no copy of data, some trivial arrays
        [Base.issingletontype(T) ? :(similar(xs, $T)) : :(reinterpret($T, xs)) for T in Ts.parameters]
    else
        [:(map($(Get(i)), xs)) for i in 1:length(fieldnames(Ts))]
    end
    Expr(:tuple, each...)
end

struct Get{i} end
Get(i) = Get{Int(i)}()
(::Get{i})(x) where {i} = x[i]

function ChainRulesCore.rrule(::typeof(unzip), xs::AbstractArray{T}) where {T <: Tuple}
    function rezip(dy)
        dxs = map(unthunk.(dy)...) do ys...
            Tangent{T}(ys...)
        end
        (NoTangent(), dxs)
    end
    return unzip(xs), rezip
end
