# A functor for testing
struct Multiplier{T}
    x::T
end
(m::Multiplier)(y) = m.x * y
function ChainRulesCore.rrule(m::Multiplier, y)
    Multiplier_pullback(z̄) = Tangent{typeof(m)}(; x=y * z̄), m.x * z̄
    return m(y), Multiplier_pullback
end

"A function that outputs a vector from a scalar for testing"
make_two_vec(x) = [x, x]
    make_two_vec_pullback(ȳ) = (NoTangent(), sum(ȳ))
function ChainRulesCore.rrule(::typeof(make_two_vec), x)
    return make_two_vec(x), make_two_vec_pullback
end

@testset "test_helpers.jl" begin
    @testset "Multiplier functor" begin
        test_rrule(Multiplier(4.0), 3.0)
    end
    @testset "make_two_vec" begin
        test_rrule(make_two_vec, 1.5)
    end
end