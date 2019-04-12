using LinearAlgebra: AbstractTriangular

#####
##### We need to be able to convert everything into into a vector for use with FDM.jl
#####

# Transform `x` into a vector, and return a closure which inverts the transformation.
to_vec(x::Nothing) = (x, nothing)
to_vec(x::Real) = ([x], x->x[1])

# Arrays.
to_vec(x::Vector{<:Real}) = (x, identity)
to_vec(x::Array) = vec(x), x_vec->reshape(x_vec, size(x))

# AbstractArrays.
function to_vec(x::T) where {T<:AbstractTriangular}
    x_vec, back = to_vec(Matrix(x))
    return x_vec, x_vec->T(reshape(back(x_vec), size(x)))
end
to_vec(x::Symmetric) = vec(Matrix(x)), x_vec->Symmetric(reshape(x_vec, size(x)))
to_vec(X::Diagonal) = vec(Matrix(X)), x_vec->Diagonal(reshape(x_vec, size(X)...))

# Non-array data structures.
function to_vec(x::Tuple)
    x_vecs, x_backs = zip(map(to_vec, x)...)
    sz = cumsum([map(length, x_vecs)...])
    return vcat(x_vecs...), v->([x_backs[n](v[sz[n]-length(x[n])+1:sz[n]]) for n in 1:length(x)]...,)
end

function FDM.jvp(fdm, f, x, ẋ)
    x_vec, vec_to_x = to_vec(x)
    ẋ_vec, _ = to_vec(ẋ)
    y = f(x)
    _, vec_to_y = to_vec(y)
    return vec_to_y(FDM.jvp(fdm, x_vec->to_vec(f(vec_to_x(x_vec)))[1], x_vec, ẋ_vec))
end

function FDM.j′vp(fdm, f, ȳ, x)
    x_vec, vec_to_x = to_vec(x)
    ȳ_vec, _ = to_vec(ȳ)
    return vec_to_x(FDM.j′vp(fdm, x_vec->to_vec(f(vec_to_x(x_vec)))[1], ȳ_vec, x_vec))
end

function j′vp(fdm, f, ȳ, xs...)
    return (map(enumerate(xs)) do (p, x)
        return FDM.j′vp(
            fdm,
            function(x)
                xs_ = [xs...]
                xs_[p] = x
                return f(xs_...)
            end,
            ȳ,
            x,
        )    
    end...,)
end

# My version of isapprox
function fd_isapprox(x_ad::Nothing, x_fd, rtol, atol)
    return isapprox(x_fd, zero(x_fd); rtol=rtol, atol=atol)
end
function fd_isapprox(x_ad::AbstractArray, x_fd::AbstractArray, rtol, atol)
    return isapprox(x_ad, x_fd; rtol=rtol, atol=atol)
end
function fd_isapprox(x_ad::Real, x_fd::Real, rtol, atol)
    return isapprox(x_ad, x_fd; rtol=rtol, atol=atol)
end
function fd_isapprox(x_ad::NamedTuple, x_fd, rtol, atol)
    f = (x_ad, x_fd)->fd_isapprox(x_ad, x_fd, rtol, atol)
    return all([f(getfield(x_ad, key), getfield(x_fd, key)) for key in keys(x_ad)])
end
function fd_isapprox(x_ad::Tuple, x_fd::Tuple, rtol, atol)
    return all(map((x, x′)->fd_isapprox(x, x′, rtol, atol), x_ad, x_fd))
end

# Ensure that `to_vec` and j′vp works correctly.
for x in [
        randn(10), 5.0, randn(10, 10), (5.0, 4.0), (randn(10), randn(11)),
        Diagonal(randn(10)),
    ]
    x_vec, back = to_vec(x)
    @test x_vec isa AbstractVector{<:Real}
    @test back(x_vec) == x
    @test fd_isapprox(j′vp(central_fdm(5, 1), identity, x, x), (x,), 1e-10, 1e-10)
end

"""
    frule_test(f, x, ẋ; rtol=1e-9, atol=1e-9, fdm=central_fdm(5, 1))

# Arguments
- `f`: Function for which the `frule` should be tested.
- `x`: input at which to evaluate `f` (should generally be set randomly).
- `ẋ`: differential w.r.t. `x` (should generally be set randomly).
"""
function frule_test(f, (x, ẋ); rtol=1e-9, atol=1e-9, fdm=central_fdm(5, 1))
    return frule_test(f, ((x, ẋ),); rtol=rtol, atol=atol, fdm=fdm)
end

function frule_test(f, xẋs::Tuple{Any, Any}...; rtol=1e-9, atol=1e-9, fdm=central_fdm(5, 1))
    xs, ẋs = collect(zip(xẋs...))
    Ω, dΩ_rule = ChainRules.frule(f, xs...)
    @test f(xs...) == Ω

    dΩ_ad, dΩ_fd = dΩ_rule(ẋs...), FDM.jvp(fdm, xs->f(xs...), xs, ẋs)
    @test chain_rules_isapprox(dΩ_ad, dΩ_fd, rtol, atol)
end

"""
    rrule_test(f, ȳ, (x, x̄); rtol=1e-9, atol=1e-9, fdm=central_fdm(5, 1))

# Arguments
- `f`: Function to which rule should be applied.
- `ȳ`: adjoint w.r.t. output of `f` (should generally be set randomly).
- `x`: input at which to evaluate `f` (should generally be set randomly).
- `x̄`: currently accumulated adjoint (should generally be set randomly).
"""
function rrule_test(
    f, ȳ, (x, x̄)::Tuple{Any, Any};
    rtol=1e-9, atol=1e-9, fdm=central_fdm(5, 1),
)
    # Check correctness of evaluation.
    fx, dx = ChainRules.rrule(f, x)
    @test fx ≈ f(x)

    # Correctness testing via finite differencing.
    x̄_ad, x̄_fd = dx(ȳ), j′vp(fdm, f, ȳ, x)[1]
    @test chain_rules_isapprox(x̄_ad, x̄_fd, rtol, atol)

    # Assuming x̄_ad to be correct, check that other ChainRules mechanisms are correct.
    test_adjoint!(x̄, dx, ȳ, x̄_ad)
end

function rrule_test(
    f, ȳ, xx̄s::Tuple{Any, Any}...;
    rtol=1e-9, atol=1e-9, fdm=central_fdm(5, 1),
)
    # Check correctness of evaluation.
    xs, x̄s = collect(zip(xx̄s...))
    Ω, Δx_rules = ChainRules.rrule(f, xs...)
    @test f(xs...) == Ω

    # Correctness testing via finite differencing.
    Δxs_ad, Δxs_fd = map(Δx_rule->Δx_rule(ȳ), Δx_rules), j′vp(fdm, f, ȳ, xs...)
    @test all(map(
        (Δx_ad, Δx_fd)->chain_rules_isapprox(Δx_ad, Δx_fd, rtol, atol),
        Δxs_ad,
        Δxs_fd,
    ))

    # Assuming the above to be correct, check that other ChainRules mechanisms are correct.
    map((x̄, Δx_rule, Δx_ad)->test_adjoint!(x̄, Δx_rule, ȳ, Δx_ad), x̄s, Δx_rules, Δxs_ad)
end

function chain_rules_isapprox(d_ad, d_fd, rtol, atol)
    return isapprox(d_ad, d_fd; rtol=rtol, atol=atol)
end
function chain_rules_isapprox(ad::Wirtinger, fd, rtol, atol)
    error("Finite differencing with Wirtinger rules not implemented")
end
function chain_rules_isapprox(d_ad::Casted, d_fd, rtol, atol)
    return all(isapprox.(extern(d_ad), d_fd; rtol=rtol, atol=atol))
end
function chain_rules_isapprox(d_ad::DNE, d_fd, rtol, atol)
    error("Tried to differentiate w.r.t. a DNE")
end
function chain_rules_isapprox(d_ad::Thunk, d_fd, rtol, atol)
    return isapprox(extern(d_ad), d_fd; rtol=rtol, atol=atol)
end

function test_adjoint!(x̄, dx, ȳ, partial)
    x̄_old = copy(x̄)
    x̄_zeros = zero.(x̄)

    @test all(accumulate(Zero(), dx, ȳ) .== accumulate(x̄_zeros, dx, ȳ))
    @test all(accumulate(x̄, dx, ȳ) .== (x̄ .+ partial))
    @test x̄ == x̄_old

    accumulate!(x̄, dx, ȳ)
    @test x̄ == (x̄_old .+ partial)
    x̄ .= x̄_old

    store!(x̄, dx, ȳ)
    @test all(x̄ .== partial)
    x̄ .= x̄_old

    return nothing
end
