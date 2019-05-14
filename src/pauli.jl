module pauli

export PauliBasis, PauliTransferMatrix

import Base: ==

using ..bases, ..spin, ..superoperators

"""
    PauliBasis(N)

Basis for an N-qubit space where `N` specifices the number of qubits. The
dimension of the basis is 2²ᴺ.
"""
mutable struct PauliBasis{B<:Tuple{Vararg{Basis}}} <: Basis
    shape::Vector{Int}
    bases::B
    function PauliBasis(N::Int)
        spins = reduce(⊗, (SpinBasis(1//2) for _ in 1:N))
        return new{Tuple{(SpinBasis{1//2} for _ in 1:N)...}}(spins.shape, spins.bases)
    end
end
==(pb1::PauliBasis, pb2::PauliBasis) = length(pb1.bases) == length(pb2.bases)

abstract type PauliTransferMatrix{B1<:Tuple{PauliBasis, PauliBasis},B2<:Tuple{PauliBasis, PauliBasis}} end

mutable struct DensePauliTransferMatrix{B1<:Tuple{PauliBasis, PauliBasis}, B2<:Tuple{PauliBasis, PauliBasis}, T<:Matrix{ComplexF64}} <: PauliTransferMatrix{B1, B2}
    basis_l::B1
    basis_r::B2
    data::T
    function DensePauliTransferMatrix{BL, BR, T}(basis_l::BL, basis_r::BR, data::T) where {BL<:Tuple{PauliBasis, PauliBasis}, BR<:Tuple{PauliBasis, PauliBasis}, T<:Matrix{ComplexF64}}
        if length(basis_l[1])*length(basis_l[2]) != size(data, 1) ||
           length(basis_r[1])*length(basis_r[2]) != size(data, 2)
            throw(DimensionMismatch())
        end
        new(basis_l, basis_r, data)
    end
end

function PauliTransferMatrix(sop::SuperOperator)
    @assert czsop.basis_l[1] == czsop.basis_l[2]
    @assert czsop.basis_r[1] == czsop.basis_r[2]

    

    DensePauliTransferMatrix(sop.basis_l, sop.basis_r, )
end

# mutable struct PauliBasis <: Basis
#     N::Int
#     function PauliBasis(N::Int)
#         if N < 0
#             throw(DimensionMismatch())
#         end
#         new([2^(2N)], N)
#     end
# end


end # end module
