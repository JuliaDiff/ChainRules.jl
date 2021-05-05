@testset "Maps and Reductions" begin
    @testset "sum" begin
        sizes = (3, 4, 7)
        @testset "dims = $dims" for dims in (:, 1)
            @testset "Array{$N, $T}" for N in eachindex(sizes), T in (Float64, ComplexF64)
                x = randn(T, sizes[1:N]...)
                test_frule(sum, x; fkwargs=(;dims=dims))
                test_rrule(sum, x; fkwargs=(;dims=dims))
            end
        end
    end  # sum

    @testset "sum abs2" begin
        sizes = (3, 4, 7)
        @testset "dims = $dims" for dims in (:, 1)
            @testset "Array{$N, $T}" for N in eachindex(sizes), T in (Float64, ComplexF64)
                x = randn(T, sizes[1:N]...)
                test_frule(sum, abs2, x; fkwargs=(;dims=dims))
                test_rrule(sum, abs2 ⊢ DoesNotExist(), x; fkwargs=(;dims=dims))
            end
        end
    end  # sum abs2

    @testset "prod" begin
        @testset "Array{$T}" for T in [Float64] # [Float64, ComplexF64]
            @testset "size = $sz, dims = $dims" for (sz, dims) in [
                ((12,), :), ((12,), 1),
                ((3,4), 1), ((3,4), 2), ((3,4), :), ((3,4), [1,2]),
                ((3,4,1), 1), ((3,2,2), 3), ((3,2,2), 2:3),
                ]
                x = randn(T, sz)
                test_rrule(prod, x; fkwargs=(dims=dims,), check_inferred=true)

                x[1] = 0
                test_rrule(prod, x; fkwargs=(dims=dims,), check_inferred=true)

                x[5] = 0
                test_rrule(prod, x; fkwargs=(dims=dims,), check_inferred=true)

                x[3] = x[7] = 0  # two zeros along some slice, for any dims
                test_rrule(prod, x; fkwargs=(dims=dims,), check_inferred=true)

                if ndims(x) == 3
                    xp = PermutedDimsArray(x, (3,2,1))  # not a StridedArray
                    xpdot, xpbar = permutedims(rand(T, sz), (3,2,1)), permutedims(rand(T, sz), (3,2,1))
                    test_rrule(prod, xp ⊢ xpbar; fkwargs=(dims=dims,), check_inferred=true)
                end
            end
        end
        @testset "Array{Float32}, no zero entries" begin
            v = [1f-5, 1f-10, 1f-15, 1f-20]
            @test prod(v) == 0
            @test unthunk(rrule(prod, v)[2](1f0)[2]) == zeros(4)
            test_rrule(prod, v)
        end
    end # prod
end
