#####
##### constructors
#####

ChainRules.@non_differentiable (::Type{T} where {T<:Array})(::UndefInitializer, args...)

function frule((_, ẋ), ::Type{T}, x::AbstractArray) where {T<:Array}
    return T(x), T(ẋ)
end

function frule((_, ẋ), ::Type{AbstractArray{T}}, x::AbstractArray) where {T}
    return AbstractArray{T}(x), AbstractArray{T}(ẋ)
end

function rrule(::Type{T}, x::AbstractArray) where {T<:Array}
    project_x = ProjectTo(x)
    Array_pullback(ȳ) = (NoTangent(), project_x(ȳ))
    return T(x), Array_pullback
end

# This abstract one is used for `float(x)` and other float conversion purposes:
function rrule(::Type{AbstractArray{T}}, x::AbstractArray) where {T}
    project_x = ProjectTo(x)
    AbstractArray_pullback(ȳ) = (NoTangent(), project_x(ȳ))
    return AbstractArray{T}(x), AbstractArray_pullback
end

#####
##### `vect`
#####

@non_differentiable Base.vect()

function frule((_, ẋs...), ::typeof(Base.vect), xs...)
    return Base.vect(xs...), Base.vect(_instantiate_zeros(ẋs, xs)...)
end

# Case of uniform type `T`: the data passes straight through,
# so no projection should be required.
function rrule(::typeof(Base.vect), X::Vararg{T, N}) where {T, N}
    vect_pullback(ȳ) = (NoTangent(), NTuple{N}(ȳ)...)
    return Base.vect(X...), vect_pullback
end

# Numbers and arrays are often promoted, to make a uniform vector.
# ProjectTo here reverses this
function rrule(
    ::typeof(Base.vect),
    X::Vararg{Union{Number,AbstractArray{<:Number}}, N},
) where {N}
    projects = map(ProjectTo, X)
    function vect_pullback(ȳ)
        X̄ = ntuple(n -> projects[n](ȳ[n]), N)
        return (NoTangent(), X̄...)
    end
    return Base.vect(X...), vect_pullback
end

# Data is unmodified, so no need to project.
function rrule(::typeof(Base.vect), X::Vararg{Any,N}) where {N}
    vect_pullback(ȳ) = (NoTangent(), ntuple(n -> ȳ[n], N)...)
    return Base.vect(X...), vect_pullback
end

"""
    _instantiate_zeros(ẋs, xs)

Forward rules for `vect`, `cat` etc may receive a mixture of data and `ZeroTangent`s.
To avoid `vect(1, ZeroTangent(), 3)` or worse `vcat([1,2], ZeroTangent(), [6,7])`, this
materialises each zero `ẋ` to be `zero(x)`.
"""
_instantiate_zeros(ẋs, xs) = map(_i_zero, ẋs, xs)
_i_zero(ẋ, x) = ẋ
_i_zero(ẋ::AbstractZero, x) = zero_tangent(x)

# Fast paths. Should it also collapse all-Zero cases?
_instantiate_zeros(ẋs::Tuple{Vararg{Number}}, xs) = ẋs
_instantiate_zeros(ẋs::Tuple{Vararg{AbstractArray}}, xs) = ẋs
_instantiate_zeros(ẋs::AbstractArray{<:Number}, xs) = ẋs
_instantiate_zeros(ẋs::AbstractArray{<:AbstractArray}, xs) = ẋs

#####
##### `copyto!`
#####

function frule((_, ẏ, ẋ), ::typeof(copyto!), y::AbstractArray, x)
    if ẏ isa AbstractZero
        # it's allowed to have an imutable zero tangent for ẏ as long as ẋ is zero
        @assert iszero(ẋ)
    else
        copyto!(ẏ, ẋ)
    end
    return copyto!(y, x), ẏ
end

function frule((_, ẏ, _, ẋ), ::typeof(copyto!), y::AbstractArray, i::Integer, x, js::Integer...)
    return copyto!(y, i, x, js...), copyto!(ẏ, i, ẋ, js...)
end

#####
##### `reshape`
#####

function frule((_, ẋ), ::typeof(reshape), x::AbstractArray, dims...)
    return reshape(x, dims...), reshape(ẋ, dims...)
end

function rrule(::typeof(reshape), A::AbstractArray, dims...)
    ax = axes(A)
    project = ProjectTo(A)  # Projection is here for e.g. reshape(::Diagonal, :)
    ∂dims = broadcast(Returns(NoTangent()), dims)
    reshape_pullback(Ȳ) = (NoTangent(), project(reshape(Ȳ, ax)), ∂dims...)
    return reshape(A, dims...), reshape_pullback
end

#####
##### `dropdims`
#####

function frule((_, ẋ), ::typeof(dropdims), x::AbstractArray; dims)
    return dropdims(x; dims), dropdims(ẋ; dims)
end

function rrule(::typeof(dropdims), A::AbstractArray; dims)
    ax = axes(A)
    project = ProjectTo(A)
    dropdims_pullback(Ȳ) = (NoTangent(), project(reshape(Ȳ, ax)))
    return dropdims(A; dims), dropdims_pullback
end

#####
##### `permutedims`
#####

function frule((_, ẋ), ::typeof(permutedims), x::AbstractArray, perm...)
    return permutedims(x, perm...), permutedims(ẋ, perm...)
end

function frule((_, ẏ, ẋ), ::typeof(permutedims!), y::AbstractArray, x::AbstractArray, perm...)
    return permutedims!(y, x, perm...), permutedims!(ẏ, ẋ, perm...)
end

function frule((_, ẋ), ::Type{<:PermutedDimsArray}, x::AbstractArray, perm)
    return PermutedDimsArray(x, perm), PermutedDimsArray(ẋ, perm)
end

function rrule(::typeof(permutedims), x::AbstractVector)
    project = ProjectTo(x)
    permutedims_pullback_1(dy) = (NoTangent(), project(permutedims(unthunk(dy))))
    permutedims_pullback_1(::ZeroTangent) = (NoTangent(), ZeroTangent())
    return permutedims(x), permutedims_pullback_1
end

function rrule(::typeof(permutedims), x::AbstractArray, perm)
    pr = ProjectTo(x)  # projection restores e.g. transpose([1,2,3])
    permutedims_back_2(dy) = (NoTangent(), pr(permutedims(unthunk(dy), invperm(perm))), NoTangent())
    permutedims_back_2(::ZeroTangent) = (NoTangent(), ZeroTangent(), NoTangent())
    return permutedims(x, perm), permutedims_back_2
end

function rrule(::Type{<:PermutedDimsArray}, x::AbstractArray, perm)
    pr = ProjectTo(x)
    permutedims_back_3(dy) = (NoTangent(), pr(permutedims(unthunk(dy), invperm(perm))), NoTangent())
    permutedims_back_3(::ZeroTangent) = (NoTangent(), ZeroTangent(), NoTangent())
    return PermutedDimsArray(x, perm), permutedims_back_3
end

#####
##### `repeat`
#####

function frule((_, ẋs), ::typeof(repeat), xs::AbstractArray, cnt...; kw...)
    return repeat(xs, cnt...; kw...), repeat(ẋs, cnt...; kw...)
end

function rrule(::typeof(repeat), xs::AbstractArray; inner=nothing, outer=nothing)

    project_Xs = ProjectTo(xs)
    S = size(xs)
    inner_size = inner === nothing ? ntuple(Returns(1), ndims(xs)) : inner
    function repeat_pullback(ȳ)
        dY = unthunk(ȳ)
        Δ′ = zero(xs)
        # Loop through each element of Δ, calculate source dimensions, accumulate into Δ′
        for (dest_idx, val) in pairs(IndexCartesian(), dY)
            # First, round dest_idx[dim] to nearest gridpoint defined by inner_dims[dim], then
            # wrap around based on original size S.
            src_idx = [mod1(div(dest_idx[dim] - 1, inner_size[dim]) + 1, S[dim]) for dim in 1:length(S)]
            Δ′[src_idx...] += val
        end
        x̄ = project_Xs(Δ′)
        return (NoTangent(), x̄)
    end

    return repeat(xs; inner = inner, outer = outer), repeat_pullback
end

function rrule(::typeof(repeat), xs::AbstractArray, counts::Integer...)

    project_Xs = ProjectTo(xs)
    S = size(xs)
    function repeat_pullback(ȳ)
        dY = unthunk(ȳ)
        size2ndims = ntuple(d -> isodd(d) ? get(S, 1+d÷2, 1) : get(counts, d÷2, 1), 2*ndims(dY))
        reduced = sum(reshape(dY, size2ndims); dims = ntuple(d -> 2d, ndims(dY)))
        x̄ = project_Xs(reshape(reduced, S))
        return (NoTangent(), x̄, map(Returns(NoTangent()), counts)...)
    end
    return repeat(xs, counts...), repeat_pullback
end

#####
##### `hcat`
#####

function frule((_, ẋs...), ::typeof(hcat), xs...)
    return hcat(xs...), hcat(_instantiate_zeros(ẋs, xs)...)
end

# All the [hv]cat functions treat anything that's not an array as a scalar. 
_catsize(x) = ()
_catsize(x::AbstractArray) = size(x)

function rrule(::typeof(hcat), Xs...)
    Y = hcat(Xs...)  # note that Y always has 1-based indexing, even if X isa OffsetArray
    Base.require_one_based_indexing(Y)
    ndimsY = Val(ndims(Y))  # this avoids closing over Y, Val() is essential for type-stability
    sizes = map(_catsize, Xs)   # this avoids closing over Xs
    project_Xs = map(ProjectTo, Xs)
    function hcat_pullback(ȳ)
        dY = unthunk(ȳ)
        hi = Ref(0)  # Ref avoids hi::Core.Box
        dXs = map(project_Xs, sizes) do project, sizeX
            ndimsX = length(sizeX)
            lo = hi[] + 1
            hi[] += get(sizeX, 2, 1)
            ind = ntuple(ndimsY) do d
                if d==2
                    d > ndimsX ? lo : lo:hi[]
                else
                    d > ndimsX ? 1 : (:)
                end
            end
            InplaceableThunk(
                dX -> dX .+= view(dY, ind...),
                @thunk project(@allowscalar dY[ind...])
            )
        end
        return (NoTangent(), dXs...)
    end
    return Y, hcat_pullback
end

function frule((_, _, Ȧs), ::typeof(reduce), ::typeof(hcat), As::AbstractVector{<:AbstractVecOrMat})
    return reduce(hcat, As), reduce(hcat, _instantiate_zeros(Ȧs, As))
end

function rrule(::typeof(reduce), ::typeof(hcat), As::AbstractVector{<:AbstractVecOrMat})
    Y = reduce(hcat, As)
    Base.require_one_based_indexing(Y)
    widths = map(A -> size(A,2), As)
    function reduce_hcat_pullback_2(dY)
        hi = Ref(0)
        dAs = map(widths) do w
            lo = hi[]+1
            hi[] += w
            dY[:, lo:hi[]]
        end
        return (NoTangent(), NoTangent(), dAs)
    end
    return Y, reduce_hcat_pullback_2
end

function rrule(::typeof(reduce), ::typeof(hcat), As::AbstractVector{<:AbstractVector})
    axe = axes(As,1)
    function reduce_hcat_pullback_1(dY)
        hi = Ref(0)
        dAs = map(_ -> dY[:, hi[]+=1], axe)
        return (NoTangent(), NoTangent(), dAs)
    end
    return reduce(hcat, As), reduce_hcat_pullback_1
end

#####
##### `vcat`
#####

function frule((_, ẋs...), ::typeof(vcat), xs...)
    return vcat(xs...), vcat(_instantiate_zeros(ẋs, xs)...)
end

function rrule(::typeof(vcat), Xs...)
    Y = vcat(Xs...)
    Base.require_one_based_indexing(Y)
    ndimsY = Val(ndims(Y))
    sizes = map(_catsize, Xs)
    project_Xs = map(ProjectTo, Xs)
    function vcat_pullback(ȳ)
        dY = unthunk(ȳ)
        hi = Ref(0)
        dXs = map(project_Xs, sizes) do project, sizeX
            ndimsX = length(sizeX)
            lo = hi[] + 1
            hi[] += get(sizeX, 1, 1)
            ind = ntuple(ndimsY) do d
                if d==1
                    d > ndimsX ? lo : lo:hi[]
                else
                    d > ndimsX ? 1 : (:)
                end
            end
            InplaceableThunk(
                dX -> dX .+= view(dY, ind...),
                @thunk project(@allowscalar dY[ind...])
            )
        end
        return (NoTangent(), dXs...)
    end
    return Y, vcat_pullback
end

function frule((_, _, Ȧs), ::typeof(reduce), ::typeof(vcat), As::AbstractVector{<:AbstractVecOrMat})
    return reduce(vcat, As), reduce(vcat, _instantiate_zeros(Ȧs, As))
end

function rrule(::typeof(reduce), ::typeof(vcat), As::AbstractVector{<:AbstractVecOrMat})
    Y = reduce(vcat, As)
    Base.require_one_based_indexing(Y)
    ndimsY = Val(ndims(Y))
    heights = map(A -> size(A,1), As)
    function reduce_vcat_pullback(dY)
        hi = Ref(0)
        dAs = map(heights) do z
            lo = hi[]+1
            hi[] += z
            ind = ntuple(d -> d==1 ? (lo:hi[]) : (:), ndimsY)
            dY[ind...]
        end
        return (NoTangent(), NoTangent(), dAs)
    end
    return Y, reduce_vcat_pullback
end

#####
##### `cat`
#####

_val(::Val{x}) where {x} = x

function frule((_, ẋs...), ::typeof(cat), xs...; dims)
    return cat(xs...; dims), cat(_instantiate_zeros(ẋs, xs)...; dims)
end

function rrule(::typeof(cat), Xs...; dims)
    Y = cat(Xs...; dims=dims)
    Base.require_one_based_indexing(Y)
    _cdims = dims isa Val ? _val(dims) : dims
    cdims = _cdims isa Integer ? Int(_cdims) : Tuple(_cdims)
    ndimsY = Val(ndims(Y))
    sizes = map(_catsize, Xs)
    project_Xs = map(ProjectTo, Xs)
    function cat_pullback(ȳ)
        dY = unthunk(ȳ)
        prev = fill(0, _val(ndimsY))  # note that Y always has 1-based indexing, even if X isa OffsetArray
        dXs = map(project_Xs, sizes) do project, sizeX
            ndimsX = length(sizeX)
            index = ntuple(ndimsY) do d
                if d in cdims
                    d > ndimsX ? (prev[d]+1) : (prev[d]+1:prev[d]+sizeX[d])
                else
                    d > ndimsX ? 1 : 1:sizeX[d]
                end
            end
            for d in cdims
                prev[d] += get(sizeX, d, 1)
            end
            InplaceableThunk(
                dX -> dX .+= view(dY, index...),
                @thunk project(@allowscalar dY[index...])
            )
        end
        return (NoTangent(), dXs...)
    end
    return Y, cat_pullback
end

#####
##### `hvcat`
#####

function frule((_, _, ẋs...), ::typeof(hvcat), rows, xs...)
    return hvcat(rows, xs...), hvcat(rows, _instantiate_zeros(ẋs, xs)...)
end

function rrule(::typeof(hvcat), rows, values...)
    Y = hvcat(rows, values...)
    cols = size(Y,2)
    ndimsY = Val(ndims(Y))
    sizes = map(_catsize, values)
    project_Vs = map(ProjectTo, values)
    function hvcat_pullback(dY)
        prev = fill(0, 2)
        dXs = map(project_Vs, sizes) do project, sizeX
            ndimsX = length(sizeX)
            index = ntuple(ndimsY) do d
                if d in (1, 2)
                    d > ndimsX ? (prev[d]+1) : (prev[d]+1:prev[d]+sizeX[d])
                else
                    d > ndimsX ? 1 : (:)
                end
            end
            prev[2] += get(sizeX, 2, 1)
            if prev[2] == cols
                prev[2] = 0
                prev[1] += get(sizeX, 1, 1)
            end
            project(dY[index...])
        end
        return (NoTangent(), NoTangent(), dXs...)
    end
    return Y, hvcat_pullback
end

#####
##### `reverse`
#####

# 1-dim case allows start/stop, N-dim case takes dims keyword
# whose defaults changed in Julia 1.6... just pass them all through:

function frule((_, ẋ), ::typeof(reverse), x::Union{AbstractArray, Tuple}, args...; kw...)
    return reverse(x, args...; kw...), reverse(ẋ, args...; kw...)
end

function frule((_, ẋ), ::typeof(reverse!), x::Union{AbstractArray, Tuple}, args...; kw...)
    return reverse!(x, args...; kw...), reverse!(ẋ, args...; kw...)
end

function rrule(::typeof(reverse), x::Union{AbstractArray, Tuple}, args...; kw...)
    nots = map(Returns(NoTangent()), args)
    function reverse_pullback(dy)
        dx = @thunk reverse(unthunk(dy), args...; kw...)
        return (NoTangent(), dx, nots...)
    end
    return reverse(x, args...; kw...), reverse_pullback
end

#####
##### `circshift`
#####

function frule((_, ẋ), ::typeof(circshift), x::AbstractArray, shifts)
    return circshift(x, shifts), circshift(ẋ, shifts)
end

function frule((_, ẏ, ẋ), ::typeof(circshift!), y::AbstractArray, x::AbstractArray, shifts)
    return circshift!(y, x, shifts), circshift!(ẏ, ẋ, shifts)
end

function rrule(::typeof(circshift), x::AbstractArray, shifts)
    function circshift_pullback(dy)
        dx = @thunk circshift(unthunk(dy), map(-, shifts))
        # Note that circshift! is useless for InplaceableThunk, as it overwrites completely
        return (NoTangent(), dx, NoTangent())
    end
    return circshift(x, shifts), circshift_pullback
end

#####
##### `fill`
#####

function frule((_, ẋ), ::typeof(fill), x::Any, dims...)
    return fill(x, dims...), fill(ẋ, dims...)
end

function frule((_, ẏ, ẋ), ::typeof(fill!), y::AbstractArray, x::Any)
    return fill!(y, x), fill!(ẏ, ẋ)
end

function rrule(::typeof(fill), x::Any, dims...)
    project = ProjectTo(x)
    nots = map(Returns(NoTangent()), dims)
    fill_pullback(Ȳ) = (NoTangent(), project(sum(Ȳ)), nots...)
    return fill(x, dims...), fill_pullback
end

#####
##### `filter`
#####

function frule((_, _, ẋ), ::typeof(filter), f, x::AbstractArray)
    inds = findall(f, x)
    return x[inds], ẋ[inds]
end

function rrule(::typeof(filter), f, x::AbstractArray)
    inds = findall(f, x)
    y, back = rrule(getindex, x, inds)
    function filter_pullback(dy)
        _, dx, _ = back(dy)
        return (NoTangent(), NoTangent(), dx)
    end
    return y, filter_pullback
end

#####
##### `findmax`, `maximum`, etc.
#####

for findm in (:findmin, :findmax)
    findm_pullback = Symbol(findm, :_pullback)

    @eval function frule((_, ẋ), ::typeof($findm), x; dims=:)
        y, ind = $findm(x; dims=dims)
        return (y, ind), Tangent{typeof((y, ind))}(ẋ[ind], NoTangent())
    end

    @eval function rrule(::typeof($findm), x::AbstractArray; dims=:)
        y, ind = $findm(x; dims=dims)
        function $findm_pullback((dy, _))  # this accepts e.g. Tangent{Tuple{Float64, Int64}}(4.0, nothing)
            dy isa AbstractZero && return (NoTangent(), NoTangent())
            return (NoTangent(), thunked_∇getindex(x, dy, ind),)
        end
        return (y, ind), $findm_pullback
    end
end

# These rules for `maximum` pick the same subgradient as `findmax`:

function frule((_, ẋ), ::typeof(maximum), x; dims=:)
    y, ind = findmax(x; dims=dims)
    return y, ẋ[ind]
end

function rrule(::typeof(maximum), x::AbstractArray; dims=:)
    (y, _), back = rrule(findmax, x; dims=dims)
    maximum_pullback(dy) = back((dy, nothing))
    return y, maximum_pullback
end

function frule((_, ẋ), ::typeof(minimum), x; dims=:)
    y, ind = findmin(x; dims=dims)
    return y, ẋ[ind]
end

function rrule(::typeof(minimum), x::AbstractArray; dims=:)
    (y, _), back = rrule(findmin, x; dims=dims)
    minimum_pullback(dy) = back((dy, nothing))
    return y, minimum_pullback
end

#####
##### `extrema`
#####

function rrule(::typeof(extrema), x::AbstractArray{<:Number}; dims=:)
    if dims isa Colon
        return _extrema_colon(x)
    else
        return _extrema_dims(x, dims)
    end
end

function _extrema_colon(x)
    ylo, ilo = findmin(x)
    yhi, ihi = findmax(x)
    project = ProjectTo(x)
    function extrema_pullback((dylo, dyhi))  # accepts Tangent
        if (dylo, dyhi) isa Tuple{AbstractZero, AbstractZero}
            return (NoTangent(), NoTangent())
        end
        # One argument may be AbstractZero here. Use promote_op because 
        # promote_type allows for * as well as +, hence gives Any.
        T = Base.promote_op(+, typeof(dylo), typeof(dyhi))
        x_nothunk = let
        # x_thunk = @thunk begin  # this doesn't infer
            dx = fill!(similar(x, T, axes(x)), false)
            view(dx, ilo) .= dylo
            view(dx, ihi) .= view(dx, ihi) .+ dyhi
            project(dx)
        end
        # x_ithunk = InplaceableThunk(x_thunk) do dx
        #     view(dx, ilo) .= view(dx, ilo) .+ dylo
        #     view(dx, ihi) .= view(dx, ihi) .+ dyhi
        #     dx
        # end
        return (NoTangent(), x_nothunk)
    end
    return (ylo, yhi), extrema_pullback
end

function _extrema_dims(x, dims)
    ylo, ilo = findmin(x; dims=dims)
    yhi, ihi = findmax(x; dims=dims)
    y = similar(ylo, Tuple{eltype(ylo), eltype(yhi)})
    map!(tuple, y, ylo, yhi)  # this is a GPU-friendly version of collect(zip(ylo, yhi))
    project = ProjectTo(x)
    function extrema_pullback_dims(dy_raw)
        dy = unthunk(dy_raw)
        @assert dy isa AbstractArray{<:Tuple{Any,Any}}
        # Can we actually get Array{Tuple{Float64,ZeroTangent}} here? Not sure.
        T = Base.promote_op(+, eltype(dy).parameters...)
        x_nothunk = let
        # x_thunk = @thunk begin  # this doesn't infer
            dx = fill!(similar(x, T, axes(x)), false)
            view(dx, ilo) .= first.(dy)
            view(dx, ihi) .= view(dx, ihi) .+ last.(dy)
            project(dx)
        end
        # x_ithunk = InplaceableThunk(x_thunk) do dx
        #     view(dx, ilo) .= first.(dy)
        #     view(dx, ihi) .= view(dx, ihi) .+ last.(dy)
        #     dx
        # end
        return (NoTangent(), x_nothunk)
    end
    return y, extrema_pullback_dims
end

#####
##### `stack`
#####

function frule((_, ẋ), ::typeof(stack), x; dims::Union{Integer, Colon} = :)
    return stack(x; dims), stack(ẋ; dims)
end

# Other iterable X also allowed, maybe this should be wider?
function rrule(::typeof(stack), X::AbstractArray; dims::Union{Integer, Colon} = :)
    Y = stack(X; dims)
    sdims = if dims isa Colon
        N = ndims(Y) - ndims(X)
        X isa AbstractVector ? ndims(Y) : ntuple(i -> i + N, ndims(X))
    else
        dims
    end
    project = ProjectTo(X)
    function stack_pullback(Δ)
        dY = unthunk(Δ)
        dY isa AbstractZero && return (NoTangent(), dY)
        dX = collect(eachslice(dY; dims = sdims))
        return (NoTangent(), project(reshape(dX, project.axes)))
    end
    return Y, stack_pullback
end
