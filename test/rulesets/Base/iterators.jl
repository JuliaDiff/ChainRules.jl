
@testset "Comprehension" begin
    @testset "simple" begin
        y1, bk1 = rrule(CFG, collect, (i^2 for i in [1.0, 2.0, 3.0]))
        @test y1 == [1,4,9]
        t1 = bk1(4:6)[2]
        @test t1 isa Tangent{<:Base.Generator}
        @test t1.f == NoTangent()
        @test t1.iter ≈ 2 .* (1:3) .* (4:6)
    
        y2, bk2 = rrule(CFG, collect, Iterators.map(Counter(), [11, 12, 13.0]))
        @test y2 == map(Counter(), 11:13)
        @test bk2(ones(3))[2].iter == [93, 83, 73]
    end
end

@testset "Iterators" begin
    @testset "zip" begin
        test_rrule(collect∘zip, rand(3), rand(3))
        test_rrule(collect∘zip, rand(2,2), rand(2,2), rand(2,2))
        test_rrule(collect∘zip, rand(4), rand(2,2))

        test_rrule(collect∘zip, rand(3), rand(5))
        test_rrule(collect∘zip, rand(3,2), rand(5))
    end
end