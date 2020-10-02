# Use BLAS.gemm for strided matrix-matrix multiplication sensitivites.
const RS = StridedMatrix{<:Number}
const RST = Transpose{<:Number, <:RS}
const RSA = Adjoint{<:Number, <:RS}

# Note: weird spacing here is intentional to make this readable as a table
for (TA,  TB,  tCA, tDA, CA, DA, tCB, tDB, CB, DB) in [
    (RS,  RS,  'N', 'C', :Ȳ, :B, 'C', 'N', :A, :Ȳ),
    (RST, RS,  'N', 'T', :B, :Ȳ, 'N', 'N', :A, :Ȳ),
    (RS,  RST, 'N', 'N', :Ȳ, :B, 'T', 'N', :Ȳ, :A),
    (RST, RST, 'T', 'T', :B, :Ȳ, 'T', 'T', :Ȳ, :A),
    (RSA, RS,  'N', 'C', :B, :Ȳ, 'N', 'N', :A, :Ȳ),
    (RS,  RSA, 'N', 'N', :Ȳ, :B, 'C', 'N', :Ȳ, :A),
    (RSA, RSA, 'C', 'C', :B, :Ȳ, 'C', 'C', :Ȳ, :A),
]
    @eval function rrule(::typeof(*), A::$TA, B::$TB)
        function strided_matmul_pullback(Ȳ)
            @show :A=>($tCA, $tDA, $CA, $DA)
            @show :B=>($tCB, $tDB, $CB, $DB)
            # TODO: I  think we are messing up what is transposed for GEMM
            Ā = LinearAlgebra.BLAS.gemm($tCA, $tDA, $CA, $DA)
            B̄ = LinearAlgebra.BLAS.gemm($tCB, $tDB, $CB, $DB)
            #==
            Ā = InplaceableThunk(
                @thunk(LinearAlgebra.BLAS.gemm($tCA, $tDA, $CA, $DA)),
                X̄ -> LinearAlgebra.BLAS.gemm!($tCA, $tDA, 1.0, $CA, $DA, 1.0, X̄),
            )
            B̄ = InplaceableThunk(
                @thunk(LinearAlgebra.BLAS.gemm($tCB, $tDB, $CB, $DB)),
                X̄ -> LinearAlgebra.BLAS.gemm!($tCB, $tDB, 1.0, $CB, $DB, 1.0, X̄),
            )
            ==#
            return (NO_FIELDS, Ā, B̄)
        end
        return A*B, strided_matmul_pullback
    end
end
