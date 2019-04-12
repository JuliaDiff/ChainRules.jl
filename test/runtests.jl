# TODO: more tests!

using ChainRules, Test, FDM, LinearAlgebra, Random
using ChainRules: rrule, frule, extern, accumulate, accumulate!, store!, @scalar_rule,
    Wirtinger, wirtinger_primal, wirtinger_conjugate, add_wirtinger, mul_wirtinger,
    Zero, add_zero, mul_zero, One, add_one, mul_one, Casted, cast, add_casted, mul_casted,
    DNE, Thunk, Casted, Wirtinger
using Base.Broadcast: broadcastable

include("test_util.jl")

# @testset "Misc. Tests" begin

#     #####
#     ##### `*(x, y)`
#     #####

#     x, y = rand(3, 2), rand(2, 5)
#     z, (dx, dy) = rrule(*, x, y)

#     @test z == x * y

#     z̄ = rand(3, 5)

#     @test dx(z̄) == extern(accumulate(zeros(3, 2), dx, z̄))
#     @test dy(z̄) == extern(accumulate(zeros(2, 5), dy, z̄))

#     test_adjoint!(rand(3, 2), dx, z̄, z̄ * y')
#     test_adjoint!(rand(2, 5), dy, z̄, x' * z̄)

#     #####
#     ##### `sin.(x)`
#     #####

#     x = rand(3, 3)
#     y, (dsin, dx) = rrule(broadcast, sin, x)

#     @test y == sin.(x)
#     @test extern(dx(One())) == cos.(x)

#     x̄, ȳ = rand(), rand()
#     @test extern(accumulate(x̄, dx, ȳ)) == x̄ .+ ȳ .* cos.(x)

#     x̄, ȳ = Zero(), rand(3, 3)
#     @test extern(accumulate(x̄, dx, ȳ)) == ȳ .* cos.(x)

#     x̄, ȳ = Zero(), cast(rand(3, 3))
#     @test extern(accumulate(x̄, dx, ȳ)) == extern(ȳ) .* cos.(x)

#     #####
#     ##### `hypot(x, y)`
#     #####

#     x, y = rand(2)
#     h, dxy = frule(hypot, x, y)

#     @test extern(dxy(One(), Zero())) === y / h
#     @test extern(dxy(Zero(), One())) === x / h

#     cx, cy = cast((One(), Zero())), cast((Zero(), One()))
#     dx, dy = extern(dxy(cx, cy))
#     @test dx === y / h
#     @test dy === x / h

#     cx, cy = cast((rand(), Zero())), cast((Zero(), rand()))
#     dx, dy = extern(dxy(cx, cy))
#     @test dx === y / h * cx.value[1]
#     @test dy === x / h * cy.value[2]
# end

@testset "ChainRules" begin
    # include("differentials.jl")
    # include("rules.jl")
    @testset "rules" begin
        # include(joinpath("rules", "base.jl"))
        # include(joinpath("rules", "broadcast.jl"))
        include(joinpath("rules", "linalg.jl"))
        # include(joinpath("rules", "blas.jl"))
        # include(joinpath("rules", "nanmath.jl"))
        # include(joinpath("rules", "specialfunctions.jl"))
    end
end
