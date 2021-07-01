@non_differentiable Core.print(::Any...)
@non_differentiable Core.println(::Any...)

@non_differentiable Core.show(::Any)
@non_differentiable Core.show(::Any, ::Any)

@non_differentiable Core.apply_type(::Any...)
@non_differentiable Core.typeof(::Any)

frule((_, ẋ, _), ::typeof(typeassert), x, T) = (typeassert(x, T), ẋ)
function rrule(::typeof(typeassert), x, T)
    typeassert(x, T), Δ->(NoTangent(), Δ, NoTangent())
end

frule((_, _, ȧ, ḃ), ::typeof(ifelse), c, a, b) = (ifelse(c, a, b), ifelse(c, ȧ, ḃ))
function rrule(::typeof(ifelse), c, a, b)
    ifelse(c, a, b), Δ->(NoTangent(), NoTangent(), ifelse(c, Δ, ZeroTangent()), ifelse(c, ZeroTangent(), Δ))
end