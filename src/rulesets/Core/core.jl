@non_differentiable Core.print(::Any...)
@non_differentiable Core.println(::Any...)

@non_differentiable Core.show(::Any)
@non_differentiable Core.show(::Any, ::Any)

@non_differentiable Core.isdefined(::Any, ::Any)
@non_differentiable Core.:(<:)(::Any, ::Any)

@non_differentiable Core.apply_type(::Any, ::Any...)
@non_differentiable Core.typeof(::Any)

if isdefined(Core, :_typevar)
    @non_differentiable Core._typevar(::Any...)
end
@non_differentiable TypeVar(::Any...)
@non_differentiable UnionAll(::Any, ::Any)

frule((_, ẋ, _), ::typeof(typeassert), x, T) = (typeassert(x, T), ẋ)
function rrule(::typeof(typeassert), x, T)
    typeassert_pullback(Δ) = (NoTangent(), Δ, NoTangent())
    return typeassert(x, T), typeassert_pullback
end

frule((_, _, ȧ, ḃ), ::typeof(ifelse), c, a, b) = (ifelse(c, a, b), ifelse(c, ȧ, ḃ))
function rrule(::typeof(ifelse), c, a, b)
    ifelse_pullback(Δ) = (NoTangent(), NoTangent(), ifelse(c, Δ, ZeroTangent()), ifelse(c, ZeroTangent(), Δ))
    return ifelse(c, a, b), ifelse_pullback
end
# ensure type stability for numbers
function rrule(::typeof(ifelse), c, a::Number, b::Number)
    ifelse_pullback(Δ) = (NoTangent(), NoTangent(), ifelse(c, Δ, zero(Δ)), ifelse(c, zero(Δ), Δ))
    return ifelse(c, a, b), ifelse_pullback
end

function rrule(
    ::typeof(Core._apply_iterate), ::typeof(iterate), f::F, args...) where F
    # flatten nested arguments
    flat = []
    for a in args
        push!(flat, a...)
    end
    # apply rrule of the function on the flat arguments
    y, pb = rrule(f, flat...)
    sizes = map(length, args)
    function _apply_iterate_pullback(dy)
        flat_dargs = pb(dy)
        df = flat_dargs[1]
        # group derivatives to tuples of the same sizes as arguments
        dargs = []
        j = 2
        for i=1:length(args)
            darg_val = flat_dargs[j:j + sizes[i] - 1]
            if length(darg_val) == 1 && darg_val[1] isa NoTangent
                push!(dargs, darg_val[1])
            else
                darg = Tangent{typeof(darg_val)}(darg_val...)
                push!(dargs, darg)
            end
            j = j + sizes[i]
        end
        return NoTangent(), NoTangent(), df, dargs...
    end
    return y, _apply_iterate_pullback
end