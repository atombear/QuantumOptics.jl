module states

export StateVector, Bra, Ket, length, basis, dagger, tensor,
    norm, normalize, normalize!, permutesystems, basisstate

import Base: ==, +, -, *, /, length, copy, Broadcast
import LinearAlgebra: norm, normalize, normalize!
import ..bases: basis, tensor, permutesystems, check_multiplicable, samebases

using ..bases
using LinearAlgebra


"""
Abstract base class for [`Bra`](@ref) and [`Ket`](@ref) states.

The state vector class stores the coefficients of an abstract state
in respect to a certain basis. These coefficients are stored in the
`data` field and the basis is defined in the `basis`
field.
"""
abstract type StateVector{B<:Basis,T<:Vector{ComplexF64}} end

"""
    Bra(b::Basis[, data])

Bra state defined by coefficients in respect to the basis.
"""
mutable struct Bra{B<:Basis,T<:Vector{ComplexF64}} <: StateVector{B,T}
    basis::B
    data::T
    function Bra{B,T}(b::B, data::T) where {B<:Basis,T<:Vector{ComplexF64}}
        if length(b) != length(data)
            throw(DimensionMismatch())
        end
        new(b, data)
    end
end

"""
    Ket(b::Basis[, data])

Ket state defined by coefficients in respect to the given basis.
"""
mutable struct Ket{B<:Basis,T<:Vector{ComplexF64}} <: StateVector{B,T}
    basis::B
    data::T
    function Ket{B,T}(b::B, data::T) where {B<:Basis,T<:Vector{ComplexF64}}
        if length(b) != length(data)
            throw(DimensionMismatch())
        end
        new(b, data)
    end
end

Bra{B}(b::B, data::T) where {B<:Basis,T<:Vector{ComplexF64}} = Bra{B,T}(b, data)
Ket{B}(b::B, data::T) where {B<:Basis,T<:Vector{ComplexF64}} = Ket{B,T}(b, data)

Bra(b::B, data::T) where {B<:Basis,T<:Vector{ComplexF64}} = Bra{B,T}(b, data)
Ket(b::B, data::T) where {B<:Basis,T<:Vector{ComplexF64}} = Ket{B,T}(b, data)

Bra{B}(b::B) where B<:Basis = Bra{B}(b, zeros(ComplexF64, length(b)))
Ket{B}(b::B) where B<:Basis = Ket{B}(b, zeros(ComplexF64, length(b)))
Bra(b::Basis) = Bra(b, zeros(ComplexF64, length(b)))
Ket(b::Basis) = Ket(b, zeros(ComplexF64, length(b)))

Ket(b::Basis, data) = Ket(b, convert(Vector{ComplexF64}, data))
Bra(b::Basis, data) = Bra(b, convert(Vector{ComplexF64}, data))

copy(a::T) where {T<:StateVector} = T(a.basis, copy(a.data))
length(a::StateVector) = length(a.basis)::Int
basis(a::StateVector) = a.basis

==(x::T, y::T) where {T<:Ket} = samebases(x, y) && x.data==y.data
==(x::T, y::T) where {T<:Bra} = samebases(x, y) && x.data==y.data
==(x::Ket, y::Ket) = false
==(x::Bra, y::Bra) = false

# Arithmetic operations
+(a::Ket{B}, b::Ket{B}) where {B<:Basis} = Ket(a.basis, a.data+b.data)
+(a::Bra{B}, b::Bra{B}) where {B<:Basis} = Bra(a.basis, a.data+b.data)
+(a::Ket, b::Ket) = throw(bases.IncompatibleBases())
+(a::Bra, b::Bra) = throw(bases.IncompatibleBases())

-(a::Ket{B}, b::Ket{B}) where {B<:Basis} = Ket(a.basis, a.data-b.data)
-(a::Bra{B}, b::Bra{B}) where {B<:Basis} = Bra(a.basis, a.data-b.data)
-(a::Ket, b::Ket) = throw(bases.IncompatibleBases())
-(a::Bra, b::Bra) = throw(bases.IncompatibleBases())

-(a::T) where {T<:StateVector} = T(a.basis, -a.data)

*(a::Bra{B,D}, b::Ket{B,D}) where {B<:Basis,D<:Vector{ComplexF64}} = transpose(a.data)*b.data
*(a::Bra, b::Ket) = throw(bases.IncompatibleBases())
*(a::Number, b::T) where {T<:StateVector} = T(b.basis, a*b.data)
*(a::T, b::Number) where {T<:StateVector} = T(a.basis, b*a.data)

/(a::T, b::Number) where {T<:StateVector} = T(a.basis, a.data/b)


"""
    dagger(x)

Hermitian conjugate.
"""
dagger(x::Bra) = Ket(x.basis, conj(x.data))
dagger(x::Ket) = Bra(x.basis, conj(x.data))

"""
    tensor(x::Ket, y::Ket, z::Ket...)

Tensor product ``|x⟩⊗|y⟩⊗|z⟩⊗…`` of the given states.
"""
tensor(a::Ket, b::Ket) = Ket(tensor(a.basis, b.basis), kron(b.data, a.data))
tensor(a::Bra, b::Bra) = Bra(tensor(a.basis, b.basis), kron(b.data, a.data))
tensor(state::StateVector) = state
tensor(states::Ket...) = reduce(tensor, states)
tensor(states::Bra...) = reduce(tensor, states)
tensor(states::Vector{T}) where T<:StateVector = reduce(tensor, states)

# Normalization functions
"""
    norm(x::StateVector)

Norm of the given bra or ket state.
"""
norm(x::StateVector) = norm(x.data)
"""
    normalize(x::StateVector)

Return the normalized state so that `norm(x)` is one.
"""
normalize(x::StateVector) = x/norm(x)
"""
    normalize!(x::StateVector)

In-place normalization of the given bra or ket so that `norm(x)` is one.
"""
normalize!(x::StateVector) = (rmul!(x.data, 1.0/norm(x)); nothing)

function permutesystems(state::T, perm::Vector{Int}) where T<:Ket
    @assert length(state.basis.bases) == length(perm)
    @assert isperm(perm)
    data = reshape(state.data, state.basis.shape...)
    data = permutedims(data, perm)
    data = reshape(data, length(data))
    Ket(permutesystems(state.basis, perm), data)
end
function permutesystems(state::T, perm::Vector{Int}) where T<:Bra
    @assert length(state.basis.bases) == length(perm)
    @assert isperm(perm)
    data = reshape(state.data, state.basis.shape...)
    data = permutedims(data, perm)
    data = reshape(data, length(data))
    Bra(permutesystems(state.basis, perm), data)
end

# Creation of basis states.
"""
    basisstate(b, index)

Basis vector specified by `index` as ket state.

For a composite system `index` can be a vector which then creates a tensor
product state ``|i_1⟩⊗|i_2⟩⊗…⊗|i_n⟩`` of the corresponding basis states.
"""
function basisstate(b::Basis, indices::Vector{Int})
    @assert length(b.shape) == length(indices)
    x = zeros(ComplexF64, length(b))
    x[LinearIndices(tuple(b.shape...))[indices...]] = Complex(1.)
    Ket(b, x)
end

function basisstate(b::Basis, index::Int)
    data = zeros(ComplexF64, length(b))
    data[index] = Complex(1.)
    Ket(b, data)
end


# Helper functions to check validity of arguments
function check_multiplicable(a::Bra, b::Ket)
    if a.basis != b.basis
        throw(IncompatibleBases())
    end
end

samebases(a::T, b::T) where {T<:StateVector} = samebases(a.basis, b.basis)::Bool

# Array-like functions
Base.size(x::StateVector) = size(x.data)
@inline Base.axes(x::StateVector) = axes(x.data)
Base.ndims(x::StateVector) = 1
Base.ndims(::Type{<:StateVector}) = 1
# Indexing
Base.getindex(x::StateVector, i::Int) = getindex(x.data, i)
Base.getindex(x::StateVector, I::Vararg{Int,N}) where N = getindex(x.data, I...)
Base.getindex(x::StateVector, I::CartesianIndex) = getindex(x.data, I)
Base.setindex!(x::StateVector, v, i::Int) = setindex!(x.data, v, i)
Base.setindex!(x::StateVector, v, I::Vararg{Int,N}) where N = setindex!(x.data, v, I...)
Base.setindex!(x::StateVector, v, I::CartesianIndex) = setindex!(x.data, v, I)
# Broadcasting
Base.broadcastable(x::StateVector) = x

# Custom broadcasting style
abstract type StateVectorStyle{B<:Basis} <: Broadcast.BroadcastStyle end
struct KetStyle{B<:Basis} <: StateVectorStyle{B} end
struct BraStyle{B<:Basis} <: StateVectorStyle{B} end

# Style precedence rules
Broadcast.BroadcastStyle(::Type{<:Ket{B}}) where {B<:Basis} = KetStyle{B}()
Broadcast.BroadcastStyle(::Type{<:Bra{B}}) where {B<:Basis} = BraStyle{B}()
Broadcast.BroadcastStyle(::T, ::Broadcast.DefaultArrayStyle{0}) where T<:StateVectorStyle = T() # Broadcasting with numbers
Broadcast.BroadcastStyle(::KetStyle{B1}, ::KetStyle{B2}) where {B1<:Basis,B2<:Basis} = throw(bases.IncompatibleBases())
Broadcast.BroadcastStyle(::BraStyle{B1}, ::BraStyle{B2}) where {B1<:Basis,B2<:Basis} = throw(bases.IncompatibleBases())

function Base.similar(bc::Broadcast.Broadcasted{Style}, ::Type{ElType}) where {B<:Basis,Style<:KetStyle{B},ElType<:ComplexF64}
    b = find_basis(bc)
    return Ket{B}(b, similar(Vector{ElType}, axes(bc)))
end
function Base.similar(bc::Broadcast.Broadcasted{Style}, ::Type{ElType}) where {B<:Basis,Style<:BraStyle{B},ElType<:ComplexF64}
    b = find_basis(bc)
    return Bra{B}(b, similar(Vector{ElType}, axes(bc)))
end
find_basis(bc::Broadcast.Broadcasted) = find_basis(bc.args)
find_basis(args::Tuple) = find_basis(find_basis(args[1]), Base.tail(args))
find_basis(x) = x
find_basis(a::StateVector, rest) = a.basis
find_basis(::Any, rest) = find_basis(rest)

# In-place broadcasting for Kets
@inline function Base.copyto!(dest::Ket{B}, bc::Broadcast.Broadcasted{Style}) where {B<:Basis,Style<:KetStyle{B}}
    copyto!(dest.data, convert(Broadcast.Broadcasted{Nothing}, bc))
    dest
end
@inline function Base.copyto!(dest::Ket{B}, bc::Broadcast.Broadcasted{Style,Axes,typeof(identity),Args}) where {B<:Basis,Style<:KetStyle{B},Axes,Args<:Tuple{<:Ket{B}}}
    # Performance optimization: broadcast!(identity, dest, A) is equivalent to copyto!(dest, A) if indices match
    A = bc.args[1]
    axes(dest) == axes(bc) || Base.Broadcast.throwdm(axes(dest), axes(A))
    return copyto!(dest, A)
end
Base.copyto!(dest::Ket{B1}, bc::Broadcast.Broadcasted{Style}) where {B1<:Basis,B2<:Basis,Style<:KetStyle{B2}} =
    throw(bases.IncompatibleBases())

# In-place broadcasting for Bras
@inline function Base.copyto!(dest::Bra{B}, bc::Broadcast.Broadcasted{Style}) where {B<:Basis,Style<:BraStyle{B}}
    copyto!(dest.data, convert(Broadcast.Broadcasted{Nothing}, bc))
    dest
end
@inline function Base.copyto!(dest::Bra{B}, bc::Broadcast.Broadcasted{Style,Axes,typeof(identity),Args}) where {B<:Basis,Style<:BraStyle{B},Axes,Args<:Tuple{<:Bra{B}}}
    # Performance optimization: broadcast!(identity, dest, A) is equivalent to copyto!(dest, A) if indices match
    A = bc.args[1]
    axes(dest) == axes(bc) || Base.Broadcast.throwdm(axes(dest), axes(A))
    return copyto!(dest, A)
end
Base.copyto!(dest::Bra{B1}, bc::Broadcast.Broadcasted{Style}) where {B1<:Basis,B2<:Basis,Style<:BraStyle{B2}} =
    throw(bases.IncompatibleBases())

@inline Base.copyto!(A::T,B::T) where T<:StateVector = (copyto!(A.data,B.data); A)

end # module
