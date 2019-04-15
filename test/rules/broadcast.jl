@testset "broadcast" begin
    @testset "Misc. Tests" begin
        @testset "sin.(x)" begin
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
        end
    end
end
