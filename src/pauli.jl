module pauli

export PauliBasis, PauliTransferMatrix

import Base: ==

using ..bases, ..spin, ..superoperators
using ..operators: identityoperator, AbstractOperator
using ..operators_dense: DenseOperator
using ..spin: sigmax, sigmay, sigmaz
using SparseArrays

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

function pauli_basis_vectors(num_qubits::Int64)
    pauli_funcs = (identityoperator, sigmax, sigmay, sigmaz)
    pbv = []
    for paulis in Iterators.product((pauli_funcs for _ in 1:num_qubits)...)
        basis_vector = sparse(reshape(reduce(⊗, f(SpinBasis(1//2)) for f in paulis).data, 2 ^ (2*num_qubits)))
        push!(pbv, basis_vector)
    end
    return reduce((x, y) -> [x y], pbv)
end

function PauliTransferMatrix(sop::DenseSuperOperator{B, B, Array{Complex{Float64}, 2}}) where B <: Tuple{PauliBasis, PauliBasis}
    num_qubits = length(sop.basis_l[1].bases)
    pbv = pauli_basis_vectors(num_qubits)
    sop_dim = 2 ^ (2 * num_qubits)
    data = Array{Float64}(undef, (sop_dim, sop_dim))
    for (idx, jdx) in Iterators.product(1:sop_dim, 1:sop_dim)
        data[idx, jdx] = pbv[:, idx]' * sop.data * pbv[:, jdx] / √sop_dim |> real
    end
    return DensePauliTransferMatrix(sop.basis_l, sop.basis_r, data)
end

PauliTransferMatrix(unitary::DenseOperator{B, B, Array{Complex{Float64},2}}) where B <: PauliBasis = PauliTransferMatrix(SuperOperator(unitary))

end # end module
