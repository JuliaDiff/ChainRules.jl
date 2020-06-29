@testset "Maps and Reductions" begin
    @testset "sum" begin
        @testset for T in (Float64, ComplexF64)
            @testset "Vector" begin
                M = 3
                frule_test(sum, (randn(T, M), randn(T, M)))
                rrule_test(sum, randn(T), (randn(T, M), randn(T, M)))
            end
            @testset "Matrix" begin
                M, N = 3, 4
                frule_test(sum, (randn(T, M, N), randn(T, M, N)))
                rrule_test(sum, randn(T), (randn(T, M, N), randn(T, M, N)))
            end
            @testset "Array{T, 3}" begin
                M, N, P = 3, 7, 11
                frule_test(sum, (randn(T, M, N, P), randn(T, M, N, P)))
                rrule_test(sum, randn(T), (randn(T, M, N, P), randn(T, M, N, P)))
            end
            @testset "keyword arguments" begin
                n = 4
                rrule_test(sum, randn(T, n, 1), (randn(T, n, n+1), randn(T, n, n+1)); fkwargs=(dims=2,))
            end
        end
    end  # sum

    @testset "sum abs2" begin
        @testset for T in (Float64, ComplexF64)
            @testset "Vector" begin
                M = 3
                rrule_test(sum, randn(), (abs2, nothing), (randn(T, M), randn(T, M)))
            end
            @testset "Matrix" begin
                M, N = 3, 4
                rrule_test(sum, randn(), (abs2, nothing), (randn(T, M, N), randn(T, M, N)))
            end
            @testset "Array{T, 3}" begin
                M, N, P = 3, 7, 11
                rrule_test(sum, randn(), (abs2, nothing), (randn(T, M, N, P), randn(T, M, N, P)))
            end
            @testset "keyword arguments" begin
                n = 4
                rrule_test(
                    sum,
                    randn(n, 1),
                    (abs2, nothing),
                    (randn(T, n, n+1), randn(T, n, n+1));
                    fkwargs=(dims=2,),
                )
            end
        end
    end  # sum abs2
end
