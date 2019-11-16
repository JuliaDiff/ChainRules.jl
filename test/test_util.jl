using FiniteDifferences, Test
using FiniteDifferences: jvp, j′vp
using ChainRules
using ChainRulesCore: AbstractDifferential

const _fdm = central_fdm(5, 1)


"""
    test_scalar(f, x; rtol=1e-9, atol=1e-9, fdm=central_fdm(5, 1), test_wirtinger=x isa Complex, kwargs...)

Given a function `f` with scalar input an scalar output, perform finite differencing checks,
at input point `x` to confirm that there are correct ChainRules provided.

# Arguments
- `f`: Function for which the `frule` and `rrule` should be tested.
- `x`: input at which to evaluate `f` (should generally be set to an arbitary point in the domain).

- `test_wirtinger`: test whether the wirtinger derivative is correct, too

All keyword arguments except for `fdm` and `test_wirtinger` are passed to `isapprox`.
"""
function test_scalar(f, x; rtol=1e-9, atol=1e-9, fdm=_fdm, test_wirtinger=x isa Complex, kwargs...)
    ensure_not_running_on_functor(f, "test_scalar")

    @testset "$f at $x, $(nameof(rule))" for rule in (rrule, frule)
        res = rule(f, x)
        @test res !== nothing  # Check the rule was defined
        fx,  prop_rule = res
        @test fx == f(x)  # Check we still get the normal value, right

        if rule == rrule
            ∂self, ∂x = prop_rule(1)
            @test ∂self === NO_FIELDS
        else # rule == frule
            # Got to input extra first aguement for internals
            # But it is only a dummy since this is not a functor
            ∂x = prop_rule(NamedTuple(), 1)
        end


        # Check that we get the derivative right:
        if !test_wirtinger
            @test isapprox(
                ∂x, fdm(f, x);
                rtol=rtol, atol=atol, kwargs...
            )
        else
            # For complex arguments, also check if the wirtinger derivative is correct
            ∂Re = fdm(ϵ -> f(x + ϵ), 0)
            ∂Im = fdm(ϵ -> f(x + im*ϵ), 0)
            ∂ = 0.5(∂Re - im*∂Im)
            ∂̅ = 0.5(∂Re + im*∂Im)
            @test isapprox(
                wirtinger_primal(∂x), ∂;
                rtol=rtol, atol=atol, kwargs...
            )
            @test isapprox(
                wirtinger_conjugate(∂x), ∂̅;
                rtol=rtol, atol=atol, kwargs...
            )
        end
    end
end

function ensure_not_running_on_functor(f, name)
    # if x itself is a Type, then it is a constructor, thus not a functor.
    # This also catchs UnionAll constructors which have a `:var` and `:body` fields
    f isa Type && return

    if fieldcount(typeof(f)) > 0
        throw(ArgumentError(
            "$name cannot be used on closures/functors (such as $f)"
        ))
    end
end

"""
    frule_test(f, (x, ẋ)...; rtol=1e-9, atol=1e-9, fdm=central_fdm(5, 1), kwargs...)

# Arguments
- `f`: Function for which the `frule` should be tested.
- `x`: input at which to evaluate `f` (should generally be set to an arbitary point in the domain).
- `ẋ`: differential w.r.t. `x` (should generally be set randomly).

All keyword arguments except for `fdm` are passed to `isapprox`.
"""
function frule_test(f, (x, ẋ); rtol=1e-9, atol=1e-9, fdm=_fdm, kwargs...)
    return frule_test(f, ((x, ẋ),); rtol=rtol, atol=atol, fdm=fdm, kwargs...)
end

function frule_test(f, xẋs::Tuple{Any, Any}...; rtol=1e-9, atol=1e-9, fdm=_fdm, kwargs...)
    ensure_not_running_on_functor(f, "frule_test")
    xs, ẋs = collect(zip(xẋs...))
    Ω, pushforward = ChainRules.frule(f, xs...)
    @test f(xs...) == Ω
    dΩ_ad = pushforward(NamedTuple(), ẋs...)

    # Correctness testing via finite differencing.
    dΩ_fd = jvp(fdm, xs->f(xs...), (xs, ẋs))
    @test isapprox(
        collect(dΩ_ad),  # Use collect so can use vector equality
        collect(dΩ_fd);
        rtol=rtol,
        atol=atol,
        kwargs...
    )
end


"""
    rrule_test(f, ȳ, (x, x̄)...; rtol=1e-9, atol=1e-9, fdm=central_fdm(5, 1), kwargs...)

# Arguments
- `f`: Function to which rule should be applied.
- `ȳ`: adjoint w.r.t. output of `f` (should generally be set randomly).
  Should be same structure as `f(x)` (so if multiple returns should be a tuple)
- `x`: input at which to evaluate `f` (should generally be set to an arbitary point in the domain).
- `x̄`: currently accumulated adjoint (should generally be set randomly).

All keyword arguments except for `fdm` are passed to `isapprox`.
"""
function rrule_test(f, ȳ, (x, x̄)::Tuple{Any, Any}; rtol=1e-9, atol=1e-9, fdm=_fdm, kwargs...)
    ensure_not_running_on_functor(f, "rrule_test")

    # Check correctness of evaluation.
    fx, pullback = ChainRules.rrule(f, x)
    @test collect(fx) ≈ collect(f(x))  # use collect so can do vector equality
    (∂self, x̄_ad) = if fx isa Tuple
        # If the function returned multiple values,
        # then it must have multiple seeds for propagating backwards
        pullback(ȳ...)
    else
        pullback(ȳ)
    end

    @test ∂self === NO_FIELDS  # No internal fields
    # Correctness testing via finite differencing.
    x̄_fd = j′vp(fdm, f, ȳ, x)
    @test isapprox(x̄_ad, x̄_fd; rtol=rtol, atol=atol, kwargs...)

    # Assuming x̄_ad to be correct, check that other ChainRules mechanisms are correct.
    test_accumulation(x̄, x̄_ad)
    test_accumulation(Zero(), x̄_ad)
end

function _make_fdm_call(fdm, f, ȳ, xs, ignores)
    sig = Expr(:tuple)
    call = Expr(:call, f)
    newxs = Any[]
    arginds = Int[]
    i = 1
    for (x, ignore) in zip(xs, ignores)
        if ignore
            push!(call.args, x)
        else
            push!(call.args, Symbol(:x, i))
            push!(sig.args, Symbol(:x, i))
            push!(newxs, x)
            push!(arginds, i)
        end
        i += 1
    end
    fdexpr = :(j′vp($fdm, $sig -> $call, $ȳ, $(newxs...)))
    fd = eval(fdexpr)
    fd isa Tuple || (fd = (fd,))
    args = Any[nothing for _ in 1:length(xs)]
    for (dx, ind) in zip(fd, arginds)
        args[ind] = dx
    end
    return (args...,)
end

function rrule_test(f, ȳ, xx̄s::Tuple{Any, Any}...; rtol=1e-9, atol=1e-9, fdm=_fdm, kwargs...)
    ensure_not_running_on_functor(f, "rrule_test")

    # Check correctness of evaluation.
    xs, x̄s = collect(zip(xx̄s...))
    y, pullback = rrule(f, xs...)
    @test f(xs...) == y

    @assert !(isa(ȳ, Thunk))
    ∂s = pullback(ȳ)
    ∂self = ∂s[1]
    x̄s_ad = ∂s[2:end]
    @test ∂self === NO_FIELDS

    # Correctness testing via finite differencing.
    x̄s_fd = _make_fdm_call(fdm, f, ȳ, xs, x̄s .== nothing)
    for (x̄_ad, x̄_fd) in zip(x̄s_ad, x̄s_fd)
        if x̄_fd === nothing
            # The way we've structured the above, this tests that the rule is a DoesNotExistRule
            @test x̄_ad isa DoesNotExist
        else
            @test isapprox(x̄_ad, x̄_fd; rtol=rtol, atol=atol, kwargs...)
        end
    end

    # Assuming the above to be correct, check that other ChainRules mechanisms are correct.
    for (x̄, x̄_ad) in zip(x̄s, x̄s_ad)
        x̄ === nothing && continue
        test_accumulation(x̄, x̄_ad)
        test_accumulation(Zero(), x̄_ad)
    end
end

function Base.isapprox(ad::Wirtinger, fd; kwargs...)
    error("Finite differencing with Wirtinger rules not implemented")
end

function Base.isapprox(d_ad::DoesNotExist, d_fd; kwargs...)
    error("Tried to differentiate w.r.t. a `DoesNotExist`")
end

function Base.isapprox(d_ad::AbstractDifferential, d_fd; kwargs...)
    return isapprox(extern(d_ad), d_fd; kwargs...)
end

function test_accumulation(x̄, ∂x)
    @test all(extern(x̄ + ∂x) .≈ extern(x̄) .+ extern(∂x))
    test_accumulate(x̄, ∂x)
    test_accumulate!(x̄, ∂x)
    test_store!(x̄, ∂x)
end

function test_accumulate(x̄::Zero, ∂x)
    @test extern(accumulate(x̄, ∂x)) ≈ extern(∂x)
end

function test_accumulate(x̄::Number, ∂x)
    @test extern(accumulate(x̄, ∂x)) ≈ extern(x̄) + extern(∂x)
end

function test_accumulate(x̄::AbstractArray, ∂x)
    x̄_old = copy(x̄)
    @test all(extern(accumulate(x̄, ∂x)) .≈ (extern(x̄) .+ extern(∂x)))
    @test x̄ == x̄_old  # make sure didn't mutate x̄
end

test_accumulate!(x̄::Zero, ∂x) = nothing

function test_accumulate!(x̄::Number, ∂x)
    # This case won't have been inplace as `Number` is immutable
    @test accumulate!(x̄, ∂x) ≈ accumulate(x̄, ∂x)
end

function test_accumulate!(x̄::AbstractArray, ∂x)
    x̄_copy = copy(x̄)

    accumulate!(x̄_copy, ∂x)  # this should have actually been in-place
    @test extern(x̄_copy) ≈ (extern(x̄) .+ extern(∂x))
end

test_store!(x̄::Zero, ∂x) = nothing
test_store!(x̄::Number, ∂x) = nothing

function test_store!(x̄::AbstractArray, ∂x)
    x̄_store = copy(x̄)
    store!(x̄_store, ∂x)
    @test x̄_store ≈ extern(∂x)

    # store! is the same as `accumulate!` to a zero array
    x̄_acc = false.*x̄
    accumulate!(x̄_acc, ∂x)
    @test x̄_acc ≈ x̄_store
end
