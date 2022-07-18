@testset "Dict constructors" begin
    test_rrule(Base.Dict)
    @testset "homogeneous type" begin
        test_rrule(Base.Dict, 1 => 2.0, 2 => 4.0)
        test_rrule(Base.Dict, :a => "A", :b => "B", :c => "C")
        test_rrule(Base.Dict, (1,) => randn(1, 1), (2,) => randn(2, 2))
    end
    @testset "inhomogeneous type" begin
        test_rrule(Base.Dict, "a" => 2.0, :b => 4.0)
        test_rrule(
            Base.Dict, :a => 5.0, :b => 3f0;
            atol=1e-6, rtol=1e-6,
        ) # tolerance due to Float32.
    end
end