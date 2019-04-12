# TODO: more tests!

using ChainRules, Test
using ChainRules: rrule, frule, extern, accumulate, accumulate!, store!, @scalar_rule,
                  Wirtinger, wirtinger_primal, wirtinger_conjugate, add_wirtinger, mul_wirtinger,
                  Zero, add_zero, mul_zero, One, add_one, mul_one,
                  Casted, cast, add_casted, mul_casted
using Base.Broadcast: broadcastable

cool(x) = x + 1
@testset "frule and rrule" begin
    @test frule(cool, 1) === nothing
    @test rrule(cool, 1) === nothing
    ChainRules.@scalar_rule(Main.cool(x), one(x))
    frx, fr = frule(cool, 1)
    @test frx == 2
    @test fr(1) == 1
    rrx, rr = rrule(cool, 1)
    @test rrx == 2
    @test rr(1) == 1
end

@testset "iterating rules" begin
    _, rule = frule(+, 1)
    i = 0
    for r in rule
        @test r === rule
        i += 1
    end
    @test i == 1  # rules only iterate once, yielding themselves
end

@testset "Differentials" begin
    @testset "Wirtinger" begin
        w = Wirtinger(1+1im, 2+2im)
        @test wirtinger_primal(w) == 1+1im
        @test wirtinger_conjugate(w) == 2+2im
        @test add_wirtinger(w, w) == Wirtinger(2+2im, 4+4im)
        # TODO: other add_wirtinger methods stack overflow
        @test_throws ErrorException mul_wirtinger(w, w)
        @test_throws ErrorException extern(w)
        for x in w
            @test x === w
        end
        @test broadcastable(w) == w
        @test_throws ErrorException conj(w)
    end
    @testset "Zero" begin
        z = Zero()
        @test extern(z) === false
        @test add_zero(z, z) == z
        @test add_zero(z, 1) == 1
        @test add_zero(1, z) == 1
        @test mul_zero(z, z) == z
        @test mul_zero(z, 1) == z
        @test mul_zero(1, z) == z
        for x in z
            @test x === z
        end
        @test broadcastable(z) isa Ref{Zero}
        @test conj(z) == z
    end
    @testset "One" begin
        o = One()
        @test extern(o) === true
        @test add_one(o, o) == 2
        @test add_one(o, 1) == 2
        @test add_one(1, o) == 2
        @test mul_one(o, o) == o
        @test mul_one(o, 1) == 1
        @test mul_one(1, o) == 1
        for x in o
            @test x === o
        end
        @test broadcastable(o) isa Ref{One}
        @test conj(o) == o
    end
end

#####
##### `*(x, y)`
#####

function test_adjoint!(x̄, dx, ȳ, partial)
    x̄_old = copy(x̄)
    x̄_zeros = zero.(x̄)

    @test extern(accumulate(Zero(), dx, ȳ)) == extern(accumulate(x̄_zeros, dx, ȳ))
    @test extern(accumulate(x̄, dx, ȳ)) == (x̄ .+ partial)
    @test x̄ == x̄_old

    accumulate!(x̄, dx, ȳ)
    @test x̄ == (x̄_old .+ partial)
    x̄ .= x̄_old

    store!(x̄, dx, ȳ)
    @test x̄ == partial
    x̄ .= x̄_old

    return nothing
end

x, y = rand(3, 2), rand(2, 5)
z, (dx, dy) = rrule(*, x, y)

@test z == x * y

z̄ = rand(3, 5)

@test dx(z̄) == extern(accumulate(zeros(3, 2), dx, z̄))
@test dy(z̄) == extern(accumulate(zeros(2, 5), dy, z̄))

test_adjoint!(rand(3, 2), dx, z̄, z̄ * y')
test_adjoint!(rand(2, 5), dy, z̄, x' * z̄)

#####
##### `sin.(x)`
#####

x = rand(3, 3)
y, (dsin, dx) = rrule(broadcast, sin, x)

@test y == sin.(x)
@test extern(dx(One())) == cos.(x)

x̄, ȳ = rand(), rand()
@test extern(accumulate(x̄, dx, ȳ)) == x̄ .+ ȳ .* cos.(x)

x̄, ȳ = Zero(), rand(3, 3)
@test extern(accumulate(x̄, dx, ȳ)) == ȳ .* cos.(x)

x̄, ȳ = Zero(), cast(rand(3, 3))
@test extern(accumulate(x̄, dx, ȳ)) == extern(ȳ) .* cos.(x)

#####
##### `hypot(x, y)`
#####

x, y = rand(2)
h, dxy = frule(hypot, x, y)

@test extern(dxy(One(), Zero())) === y / h
@test extern(dxy(Zero(), One())) === x / h

cx, cy = cast((One(), Zero())), cast((Zero(), One()))
dx, dy = extern(dxy(cx, cy))
@test dx === y / h
@test dy === x / h

cx, cy = cast((rand(), Zero())), cast((Zero(), rand()))
dx, dy = extern(dxy(cx, cy))
@test dx === y / h * cx.value[1]
@test dy === x / h * cy.value[2]

#####
##### More!
#####

include(joinpath("rules", "base.jl"))
