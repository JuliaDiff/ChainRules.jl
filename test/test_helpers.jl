# import CUDA
# if CUDA.functional()
#     using CUDA  # exports CuArray, etc
# else
    @info "CUDA not functional, testing via GPUArrays"
    using GPUArrays
    GPUArrays.allowscalar(false)

    # GPUArrays provides a fake GPU array, for testing
    jl_file = normpath(joinpath(pathof(GPUArrays), "..", "..", "test", "jlarray.jl"))
    using Random  # loaded within jl_file
    include(jl_file)
    using .JLArrays
    cu = jl
    CuArray{T,N} = JLArray{T,N}
# end
@test cu(rand(3)) .+ 1 isa CuArray


"""
    @gpu_test rrule(sum, rand(3))
    @gpu_test frule((0, rand(3)), sum, rand(3))

Runs the rule as shown, and then with all arrays replaced by `GPUArray`s,
and checks that the two results agree.

NB it does not check that the rule computes derivatives correctly.
"""
macro gpu_test(ex)
    if Meta.isexpr(ex, :call) && ex.args[1] in (:rrule, :frule)
        :($_gpu_test($(ex.args...))) |> esc
    else
        error("@gpu_test only acts on one rrule(...) or frule(...) expression")
    end
end

function _gpu_test(::typeof(rrule), xs...; kw...)
    y, bk = rrule(xs...; kw...)
    y1 = one.(y)  # crude way to get input sensitivity
    dxs = bk(y1)

    gpu_y, gpu_bk = rrule(_gpu(xs)...; kw...)
    gpu_dxs = gpu_bk(one.(gpu_y))

    ChainRulesTestUtils.test_approx(gpu_dxs, _gpu(dxs))  # this contains @test
end

function _gpu_test(::typeof(frule), xdots, f::Function, xs...; kw...)
    y = frule(xdots, f, xs...; kw...)
    gpu_y = frule(_gpu(xdots), f, _gpu(xs)...; kw...)
    ChainRulesTestUtils.test_approx(gpu_y, _gpu(y))
end
function _gpu_test(::typeof(frule), f::Function, xs...; kw...)
    xdots = (NoTangent(), xs...)  # a pretty crude way to generate xdots for you
    _gpu_test(frule, xdots, f, xs...; kw...)
end

_gpu(x::AbstractArray) = CuArray(x)  # make a GPUArray
_gpu(x) = x  # ignore numbers, functions, etc.
_gpu(xs::Union{Tuple, NamedTuple}) = map(_gpu, xs)  # recurse into arguments
_gpu(x::AbstractArray{<:AbstractArray}) = map(_gpu, xs)
_gpu(x::AbstractThunk) = _gpu(unthunk(x))

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