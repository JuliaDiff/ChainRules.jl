# For the CoreLogging submodule of Base. (not to be confused with the Logging stdlib)
@testset "CoreLogging.jl" begin
    @testset "with_logger" begin
        test_rrule(
            Base.CoreLogging.with_logger,
            () -> 2.0 * 3.0,
            Base.CoreLogging.NullLogger();
            check_inferred=false,
        )
    end
end