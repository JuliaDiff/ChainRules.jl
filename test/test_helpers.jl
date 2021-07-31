"""
    Multiplier(x)

Stores a fixed `x` and multiplies by it, especially for testing
the gradient of higher order functions with respect to `x`.
```
julia> map(Multiplier(pi), [1 10 100 1000])
1×4 Matrix{Float64}:
 3.14159  31.4159  314.159  3141.59

julia> map(Multiplier(2), [1 10 100], [-1 3 -7])  # two arguments
1×3 Matrix{Int64}:
 -2  60  -1400

julia> map(Multiplier([1 2; 3 4]), ([5, 6], [7, 8]))  # x isa Matrix
([17, 39], [23, 53])
```
"""
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

"""
    Divider(x)

Stores a fixed `x` and divides by it, then squares the result.

Especially for testing the gradient of higher order functions with respect to `x`.
```
julia> map(Divider(2), [1 2 3 4 10])
1×5 Matrix{Float64}:
 0.25  1.0  2.25  4.0  25.0
```
"""
struct Divider{T<:Real}
    x::T
end
(d::Divider)(y::Real) = (y / d.x)^2

function ChainRulesCore.rrule(d::Divider, y::Real)
    Divider_pullback(dΩ) = (Tangent{typeof(d)}(; x = -2 * dΩ * y^2 / d.x^3), 2 * dΩ * y / d.x^2)
    return d(y), Divider_pullback
end

"""
    Counter()

Multiplies its input by number that increments on each call,
for testing execution order. Has a gradient `rrule` which
similarly increases by 10 each call, this is *not* the true gradient!
```
julia> map(Counter(), [1 1 1 10 10 10])
1×6 Matrix{Int64}:
 1  2  3  40  50  60
```
"""
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

# Trivial rule configurations, allowing `rrule_via_ad` with simple functions:
struct TestConfigReverse <: RuleConfig{HasReverseMode} end
ChainRulesCore.rrule_via_ad(::TestConfigReverse, f, args...) = rrule(f, args...)

struct TestConfigForwards <: RuleConfig{HasForwardsMode} end
ChainRulesCore.frule_via_ad(::TestConfigReverse, args...) = frule(args...)


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
    
    @testset "Divider" begin
        test_rrule(Divider(2.3), 4.5)
        test_rrule(Divider(0.2), -3.4)
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
