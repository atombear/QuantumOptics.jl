module pauli

export PauliBasis, PauliTransferMatrix

import Base: ==

using ..bases, ..spin, ..superoperators
using ..operators: identityoperator
using ..spin: sigmax, sigmay, sigmaz

"""
    PauliBasis(N)

Basis for an N-qubit space where `N` specifices the number of qubits. The
dimension of the basis is 2²ᴺ.
"""
mutable struct PauliBasis{B<:Tuple{Vararg{Basis}}} <: Basis
    shape::Vector{Int}
    bases::B
    function PauliBasis(N::Int)
        return new{Tuple{(SpinBasis{1//2} for _ in 1:N)...}}([2 for _ in 1:N], Tuple(SpinBasis(1//2) for _ in 1:N))
    end
end
==(pb1::PauliBasis, pb2::PauliBasis) = length(pb1.bases) == length(pb2.bases)

abstract type PauliTransferMatrix{B1<:Tuple{PauliBasis, PauliBasis}, B2<:Tuple{PauliBasis, PauliBasis}} end

mutable struct DensePauliTransferMatrix{B1<:Tuple{PauliBasis, PauliBasis},
                                        B2<:Tuple{PauliBasis, PauliBasis},
                                        T<:Matrix{Float64}} <: PauliTransferMatrix{B1, B2}
    basis_l::B1
    basis_r::B2
    data::T
    function DensePauliTransferMatrix(basis_l::BL, basis_r::BR, data::T) where {BL<:Tuple{PauliBasis, PauliBasis},
                                                                                BR<:Tuple{PauliBasis, PauliBasis},
                                                                                T<:Matrix{Float64}}
        if length(basis_l[1])*length(basis_l[2]) != size(data, 1) ||
           length(basis_r[1])*length(basis_r[2]) != size(data, 2)
            throw(DimensionMismatch())
        end
        new{BL, BR, T}(basis_l, basis_r, data)
    end
end

function PauliTransferMatrix(sop::SuperOperator)
    @assert sop.basis_l[1] == sop.basis_l[2]
    @assert sop.basis_r[1] == sop.basis_r[2]

    @assert typeof(sop.basis_l[1]) <: PauliBasis
    @assert typeof(sop.basis_r[1]) <: PauliBasis

    pauli_funcs = (identityoperator, sigmax, sigmay, sigmaz)
    pauli_basis_vectors = []
    bases = sop.basis_l[1].bases
    for paulis in Iterators.product((pauli_funcs for _ in 1:length(bases))...)
        push!(pauli_basis_vectors, reduce(⊗, f(i) for (f, i) in zip(paulis, bases)))
    end

    so_dim = 2 ^ (2 * length(bases))
    data = Array{Float64}(undef, (so_dim, so_dim))

    for (idx, u) in enumerate(pauli_basis_vectors)
        for (jdx, v) in enumerate(pauli_basis_vectors)
            data[idx, jdx] = reshape(u.data, so_dim)' * sop.data * reshape(v.data, so_dim) / √so_dim |> real
        end
    end

    return DensePauliTransferMatrix(sop.basis_l, sop.basis_r, data)
end

end # end module
