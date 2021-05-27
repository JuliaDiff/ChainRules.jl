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
                test_rrule(sum, abs2 ⊢ NoTangent(), x; fkwargs=(;dims=dims))
            end
        end
    end  # sum abs2

    @testset "prod" begin
        @testset "Array{$T}" for T in [Float64, ComplexF64]
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

            @testset "structured wrappers" begin
                # Adjoint -- like PermutedDimsArray this may actually be used
                xa = adjoint(rand(T,4,4))
                test_rrule(prod, xa ⊢ rand(T,4,4))
                test_rrule(prod, xa ⊢ rand(T,4,4), fkwargs=(dims=2,))
                @test unthunk(rrule(prod, adjoint(rand(T,3,3)))[2](1.0)[2]) isa Matrix
                @test unthunk(rrule(prod, adjoint(rand(T,3,3)), dims=1)[2](ones(1,3))[2]) isa Matrix

                # Diagonal -- a stupid thing to do, product of zeros! Shouldn't be an error though:
                @test iszero(unthunk(rrule(prod, Diagonal(rand(T,3)))[2](1.0)[2]))
                @test iszero(unthunk(rrule(prod, Diagonal(rand(T,3)), dims=1)[2](ones(1,3))[2]))
                @test unthunk(rrule(prod, Diagonal(rand(T,1)))[2](1.0)[2]) == hcat(1) # 1x1 sparse matrix
                @test unthunk(rrule(prod, Diagonal(ones(T,2)), dims=1)[2](ones(1,2))[2]) == [0 1; 1 0]

                # Triangular -- almost equally stupud
                @test iszero(unthunk(rrule(prod, UpperTriangular(rand(T,3,3)))[2](1.0)[2]))
                @test unthunk(rrule(prod, UpperTriangular(ones(T,2,2)))[2](1.0)[2]) == [0 0; 1 0]

                # Symmetric -- at least this doesn't have zeros, still an unlikely combination
                xs = Symmetric(rand(T,4,4))
                @test_skip test_rrule(prod, xs ⊢ rand(T,4,4))
                @test_skip test_rrule(prod, xs ⊢ rand(T,4,4), fkwargs=(dims=2,))
                @test unthunk(rrule(prod, Symmetric(T[1 2; -333 4]))[2](1.0)[2]) == [16 8; 8 4]
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
