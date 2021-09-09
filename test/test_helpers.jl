# A functor for testing
struct Multiplier{T}
    x::T
end
(m::Multiplier)(y) = m.x * y
(m::Multiplier)(y, z) = (m.x * y) * z

function ChainRulesCore.rrule(m::Multiplier, y)
    Multiplier_pullback(dΩ) = (Tangent{typeof(m)}(; x = dΩ * y'), m.x' * dΩ)
    return m(y), Multiplier_pullback
end
function ChainRulesCore.rrule(m::Multiplier, y, z)
    Multiplier_pullback_3(dΩ) = (
        Tangent{typeof(m)}(; x = dΩ * (y * z)'),
        m.x' * dΩ * z', 
        (m.x * y)' * dΩ,
    )
    return m(y, z), Multiplier_pullback_3
end

# A functor which counts up!
mutable struct Counter
    n::Int
    Counter(n=0) = new(n)
end
Base.:(==)(a::Counter, b::Counter) = (a.n == b.n)
(c::Counter)(x) = x * (c.n += 1) 
(c::Counter)(x, y) = (x + y) * (c.n += 1)

function ChainRulesCore.rrule(c::Counter, x)
    # False gradient, again just to record sequence of calls:
    Counter_back(dΩ) = (NoTangent(), dΩ * (c.n += 10))
    return c(x), Counter_back
end
function ChainRulesCore.rrule(c::Counter, x, y)
    Counter_back_2(dΩ) = (NoTangent(), dΩ * (c.n += 10), dΩ * (c.n += 10))
    return c(x, y), Counter_back_2
end

# NoRules - has no rules defined
struct NoRules; end

"A function that outputs a vector from a scalar for testing"
make_two_vec(x) = [x, x]
function ChainRulesCore.rrule(::typeof(make_two_vec), x)
    make_two_vec_pullback(ȳ) = (NoTangent(), sum(ȳ))
    return make_two_vec(x), make_two_vec_pullback
end

@testset "test_helpers.jl" begin
    @testset "Multiplier" begin
        # One argument
        test_rrule(Multiplier(4.0), 3.0)
        test_rrule(Multiplier(5.0 + 6im), 7.0 + 8im)
        test_rrule(Multiplier(rand(2,3)), rand(3,4))

        # Two arguments
        test_rrule(Multiplier(1.2), 3.4, 5.6)
        test_rrule(Multiplier(1.0 + 2im), 3.0 + 4im, 5.0 - 6im)
        test_rrule(Multiplier(rand(2,3)), rand(3,4), rand(4,5))
    end
    @testset "Counter" begin
        c = Counter()
        @test map(c, ones(5)) == 1:5
        @test c == Counter(5)
        y, back = rrule(c, 666)
        @test back(1) == (NoTangent(), 16)
        @test back(10) == (NoTangent(), 260)
        y2, back2 = rrule(c, 777, 888)
        @test back2(1) == (NoTangent(), 37, 47)
    end
    @testset "make_two_vec" begin
        test_rrule(make_two_vec, 1.5)
    end
end