# TODO: more tests!

using ChainRules, Test, FDM, LinearAlgebra, Random
using ChainRules: One, Zero, rrule, frule, extern, cast, accumulate, accumulate!, store!,
    Casted, Wirtinger, DNE, Thunk

include("test_util.jl")

@testset "Legacy Tests" begin

    #####
    ##### `*(x, y)`
    #####

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

end

include("test_util.jl")

@testset "ChainRules" begin
    @testset "rules" begin
        include("rules/base.jl")
        include("rules/broadcast.jl")
        include("rules/linalg.jl")
        include("rules/blas.jl")
        include("rules/nanmath.jl")
        include("rules/specialfunctions.jl")
    end
end
