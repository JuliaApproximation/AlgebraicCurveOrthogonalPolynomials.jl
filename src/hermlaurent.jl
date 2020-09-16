"""
Represents Hermitian-valued Laurent series of the form

    A[1] + A[1]' + A[2]/z + A[2]'z + A[3]/z^2 + A[3]'z^2 + …
"""

struct HermLaurent{T, Coeffs<:AbstractVector} <: AbstractQuasiVector{T}
    A::Coeffs
    function HermLaurent{T,Coeffs}(A::Coeffs) where {T,Coeffs}
        @assert ishermitian(A[1])
        new{T,Coeffs}(A)
    end
end

HermLaurent{T}(A::AbstractVector) where T = HermLaurent{T,typeof(A)}(A)
HermLaurent(A::AbstractVector{<:AbstractMatrix{T}}) where T = HermLaurent{Hermitian{Complex{T},Matrix{Complex{T}}}}(A)
HermLaurent(A::AbstractMatrix) = HermLaurent(SVector{1}((A,)))

struct UnitCircle <: Domain{ComplexF64} end
in(x::Number, ::UnitCircle) = abs(x) ≈ 1

axes(F::HermLaurent) = (Inclusion(UnitCircle()),)

function getindex(F::HermLaurent, z)
    z in axes(F,1) || throw(BoundsError())
    ret = complex(F.A[1])
    for k = 2:length(F.A)
        ret .+= F.A[k] ./ z^(k-1) .+ F.A[k]' .* z^(k-1)
    end
    Hermitian(ret)
end

checkcommutes(A::AbstractMatrix, B::AbstractMatrix) = A*B ≈ B*A || throw(ArgumentError("Do not commute"))
checkcommutes(X::HermLaurent{<:Any,<:SVector{1}}, Y::HermLaurent{<:Any,<:SVector{1}}) = checkcommutes(X.A[1], Y.A[1])

# (A + B/z + B'z)*(C + D/z + D'z) == A*C + B*D' + B'D + 1/z * (A*D + B*C) + 1/z^2 * B*D
function checkcommutes(X::HermLaurent{<:Any,<:SVector{2}}, Y::HermLaurent{<:Any,<:SVector{2}})
    A,B = X.A
    C,D = Y.A
    A*C + B*D' + B'D ≈ C*A + D*B' + D'B || throw(ArgumentError("Do not commute"))
    A*D + B*C ≈ D*A + C*B || throw(ArgumentError("Do not commute"))
    checkcommutes(B,D)
end


# (A + B/z + B'z)^2 == A^2 + B*B' + B'B + 1/z * (A*B + B*A) + 1/z^2 * B^2 + z * (B'*A + A*B') + z^2 * (B')^2
function broadcasted(::QuasiArrays.DefaultQuasiArrayStyle{1}, ::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, F::HermLaurent{<:Any,<:SVector{2}}, ::Base.RefValue{Val{2}})
    A,B = F.A
    HermLaurent(SVector(B*B'+A^2 + B'B, B*A+A*B, B^2))
end

function broadcasted(::QuasiArrays.DefaultQuasiArrayStyle{1}, ::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, F::HermLaurent{<:Any,<:SVector{3}}, ::Base.RefValue{Val{2}})
    A,B,C = F.A
    HermLaurent(SVector(
    C*C' + B*B' + A^2 + B'B + C'C,
    C*B' + B*A + A*B + B'*C, 
    C*A + B^2 + A*C, 
    C*B + B*C, 
    C^2))
end

function broadcasted(::QuasiArrays.DefaultQuasiArrayStyle{1}, ::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, F::HermLaurent{<:Any,<:SVector{2}}, ::Base.RefValue{Val{3}})
    A,B = F.A
    HermLaurent(SVector(B*B'*A + B*A*B' + A^3 + A*B*B' + A*B'*B + B'*A*B + B'*B*A, 
                    B*A^2 + B^2*B' + B*B'*B + A^2*B + A*B*A + B'*B^2,
                    B*A*B + B^2*A + A*B^2, B^3))
end

broadcasted(::QuasiArrays.DefaultQuasiArrayStyle{1}, ::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, F::HermLaurent{<:Any,<:SVector{2}}, ::Base.RefValue{Val{4}}) = (F.^2).^2


# (A + B/z + B'z)*(C + D/z + D'z) == A*C + B*D' + B'D + 1/z * (A*D + B*C) + 1/z^2 * (B*D) + …
function broadcasted(::QuasiArrays.DefaultQuasiArrayStyle{1}, ::typeof(*), X::HermLaurent{<:Any,<:SVector{2}},  Y::HermLaurent{<:Any,<:SVector{2}})
    checkcommutes(X, Y)
    A,B = X.A
    C,D = Y.A
    HermLaurent(SVector(A*C + B*D' + B'D, A*D + B*C, B*D))
end

broadcasted(::QuasiArrays.DefaultQuasiArrayStyle{1}, ::typeof(+), X::HermLaurent{<:Any,<:SVector{N}},  Y::HermLaurent{<:Any,<:SVector{N}}) where N = 
    HermLaurent(X.A .+ Y.A)

isapprox(F::HermLaurent, B::UniformScaling) = F ≈ HermLaurent(I(size(F[1],1)))
isapprox(B::UniformScaling, F::HermLaurent) = HermLaurent(I(size(F[1],1))) ≈ F
function isapprox(X::HermLaurent, Y::HermLaurent)
    nrm = zero(real(eltype(eltype(X))))
    m = min(length(X.A), length(Y.A))
    for k = 1:m
        X.A[k] ≈ Y.A[k] || return false
        nrm += norm(X.A[k])
    end
    for k = m+1:length(X.A)
        norm(X.A[k]) ≤ 10*nrm*eps() || return false
    end
    for j = m+1:length(Y.A)
        norm(Y.A[k]) ≤ 10*nrm*eps() || return false
    end
    return true
end