"""
Represents Hermitian-valued Laurent series of the form

    A[1] + A[1]' + A[2]/z + A[2]'z + A[3]/z^2 + A[3]'z^2 + …
"""

struct HermLaurent{T, Coeffs<:AbstractVector} <: AbstractQuasiVector{T}
    A::Coeffs
    function HermLaurent{T,Coeffs}(A::Coeffs) where {T,Coeffs}
        @assert isapprox(A[1], A[1]'; atol=100eps())
        new{T,Coeffs}(A)
    end
end

HermLaurent{T}(A::AbstractVector) where T = HermLaurent{T,typeof(A)}(A)
HermLaurent(A::AbstractVector{<:AbstractMatrix{T}}) where T = HermLaurent{Hermitian{Complex{T},Matrix{Complex{T}}}}(A)
HermLaurent(A::AbstractMatrix...) = HermLaurent(SVector(A))

function HermLaurent(f::Function)
    N = size(f(1.0),1)
    F = Fourier{ComplexF64}()
    M = SetindexInterlace(SMatrix{N,N,ComplexF64},fill(F,N^2)...)
    θ = axes(M,1)
    c = M \ BroadcastQuasiVector{eltype(M)}(θ -> f(exp(im*θ)), θ)
    N = Int(BlockArrays.blockcolsupport(c,1)[end])

    ret = Vector{Matrix{Float64}}(undef,N)
    ret[1] = real(reshape(c[Block(1)],2,2))
    for K = 2:N
        ret[K] = reshape(real(im*c[Block(2K-2)] + c[Block(2K-1)])/2,2,2)
    end
    HermLaurent(ret)
end
struct ComplexUnitCircle <: Domain{ComplexF64} end
in(x::Number, ::ComplexUnitCircle) = abs(x) ≈ 1

axes(F::HermLaurent) = (Inclusion(ComplexUnitCircle()),)

function getindex(F::HermLaurent, z::Number)
    z in axes(F,1) || throw(BoundsError())
    ret = complex(F.A[1])
    for k = 2:length(F.A)
        ret .+= F.A[k] ./ z^(k-1) .+ F.A[k]' .* z^(k-1)
    end
    Hermitian(ret)
end

checkcommutes(A::AbstractMatrix, B::AbstractMatrix) = A*B ≈ B*A || throw(ArgumentError("Do not commute"))
checkcommutes(X::HermLaurent{<:Any,<:SVector{1}}, Y::HermLaurent{<:Any,<:SVector{1}}) = checkcommutes(X.A[1], Y.A[1])

"""
   _mul_hermlaurent_terms(X, Y)

returns the non-positive terms of Laurent series X*Y.
This suffices to check commutation, and under the conditions X and Y commute
we can then construct the HermLaurent.

We do it on SVector just because I haven't worked out the general pattern.
"""

# (A + B/z + B'z)*(C + D/z + D'z) == A*C + B*D' + B'D + 1/z * (A*D + B*C) + 1/z^2 * B*D
function _mul_hermlaurent_terms(X::SVector{2}, Y::SVector{2})
    A,B = X
    C,D = Y
    SVector(A*C + B*D' + B'D, A*D + B*C, B*D)
end
# (A + B/z + B'z + C/z^2 + C'*z^2)*(E + F/z + F'z) == A*E + B*F' + B'F + 1/z * (A*F + B*E + C*F') + 1/z^2 * (B*F + C*E) + 1/z^3 * (C*F) + …
function _mul_hermlaurent_terms(X::SVector{3}, Y::SVector{2})
    A,B,C = X
    E,F = Y
    SVector(A*E + B*F' + B'F, A*F + B*E + C*F', B*F + C*E, C*F)
end

# (A + B/z + B'z)*(E + F/z + F'z  + G/z^2 + G'*z^2) == A*E + B*F' + B'F + 1/z * (A*F + B*E + B'G) + 1/z^2 * (A*G + B*F) + 1/z^3 * (B*G) + …
function _mul_hermlaurent_terms(X::SVector{2}, Y::SVector{3})
    A,B = X
    E,F,G = Y
    SVector(A*E + B*F' + B'F, A*F + B*E + B'G, A*G + B*F, B*G)
end

function _mul_hermlaurent_terms(X::SVector{3}, Y::SVector{3})
    A,B,C = X
    E,F,G = Y
    SVector(A*E + B*F' + B'F + C*G' + C'G, A*F + B*E  + B'G + C*F', A*G + B*F  + C*E, B*G + C*F, C*G)
end

# (A + B/z + B'z + C/z^2 + C'*z^2)*(E + F/z + F'z + G/z^2 + G*z^2) ==
#          A*E + B*F' + B'F + C*G' + C'G     +
#  1/z  * (A*F + B*E  + B'G + C*F'       )    +
# 1/z^2 * (A*G + B*F  + C*E)                            +
# 1/z^3 * (B*G + C*F) + …
# 1/z^4 * (C*G)

norm(A::HermLaurent) = norm(map(norm,A.A))

function padisapprox(X::AbstractVector, Y::AbstractVector)
    tol = 10*(norm(map(norm,X)) + norm(map(norm,Y)))*eps()
    m = min(length(X), length(Y))
    for k = 1:m
        isapprox(X[k], Y[k]; atol=tol)
    end
    for k = m+1:length(X)
        norm(X[k]) ≤ tol || return false
    end
    for k = m+1:length(Y)
        norm(Y[k]) ≤ tol || return false
    end
    return true
end


checkcommutes(X::HermLaurent, Y::HermLaurent) =
    padisapprox(_mul_hermlaurent_terms(X.A,Y.A), _mul_hermlaurent_terms(Y.A,X.A)) || throw(ArgumentError("Do not commute"))

function broadcasted(::DefaultQuasiArrayStyle{1}, ::typeof(*), X::HermLaurent,  Y::HermLaurent)
    checkcommutes(X, Y)
    HermLaurent(_mul_hermlaurent_terms(X.A, Y.A))
end


# Special cases for powers. We have commutation for free. (I didn't just use A .* B because some more cases were known)
# (A + B/z + B'z)^2 == A^2 + B*B' + B'B + 1/z * (A*B + B*A) + 1/z^2 * B^2 + z * (B'*A + A*B') + z^2 * (B')^2
function broadcasted(::DefaultQuasiArrayStyle{1}, ::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, F::HermLaurent{<:Any,<:SVector{2}}, ::Base.RefValue{Val{2}})
    A,B = F.A
    HermLaurent(B*B'+A^2 + B'B, B*A+A*B, B^2)
end

function broadcasted(::DefaultQuasiArrayStyle{1}, ::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, F::HermLaurent{<:Any,<:SVector{3}}, ::Base.RefValue{Val{2}})
    A,B,C = F.A
    HermLaurent(
    C*C' + B*B' + A^2 + B'B + C'C,
    C*B' + B*A + A*B + B'*C,
    C*A + B^2 + A*C,
    C*B + B*C,
    C^2)
end

function broadcasted(::DefaultQuasiArrayStyle{1}, ::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, F::HermLaurent{<:Any,<:SVector{2}}, ::Base.RefValue{Val{3}})
    A,B = F.A
    HermLaurent(B*B'*A + B*A*B' + A^3 + A*B*B' + A*B'*B + B'*A*B + B'*B*A,
                    B*A^2 + B^2*B' + B*B'*B + A^2*B + A*B*A + B'*B^2,
                    B*A*B + B^2*A + A*B^2, B^3)
end

broadcasted(::DefaultQuasiArrayStyle{1}, ::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, F::HermLaurent{<:Any,<:SVector{2}}, ::Base.RefValue{Val{4}}) = (F.^2).^2




padbroadcast(F, A::SVector{N}, B::SVector{N}) where N = broadcast(F, A, B)

@generated function padbroadcast(F, A::SVector{N}, B::SVector{M}) where {N,M}
    if N < M
        :(vcat(broadcast(F, A, B[SOneTo($N)]), broadcast.(F, zero(eltype(eltype(A))), reverse(reverse(B)[SOneTo($(M-N))]))))
    else # M < N
        :(vcat(broadcast(F, A[SOneTo($M)], B), broadcast.(F, reverse(reverse(A)[SOneTo($(N-M))]), zero(eltype(eltype(B))))))
    end
end

for op in (:+, :-)
    @eval begin
        broadcasted(::DefaultQuasiArrayStyle{1}, ::typeof($op), X::HermLaurent,  Y::HermLaurent) =
            HermLaurent(padbroadcast($op, X.A, Y.A))

        $op(A::UniformScaling, B::HermLaurent) = $op(HermLaurent(A(size(B[1],1))), B)
        $op(A::HermLaurent, B::UniformScaling) = $op(A, HermLaurent(B(size(A[1],1))))
    end
end

for op in (:*, :/)
    @eval broadcasted(::DefaultQuasiArrayStyle{1}, ::typeof($op), X::HermLaurent, c::Number) = HermLaurent(broadcast($op, X.A, c))
end
for op in (:*, :\)
    @eval broadcasted(::DefaultQuasiArrayStyle{1}, ::typeof($op), c::Number, X::HermLaurent) = HermLaurent(broadcast($op, c, X.A))
end

isapprox(F::HermLaurent, B::UniformScaling) = F ≈ HermLaurent(B(size(F[1],1)))
isapprox(B::UniformScaling, F::HermLaurent) = HermLaurent(I(size(F[1],1))) ≈ F

isapprox(X::HermLaurent, Y::HermLaurent) = padisapprox(X.A, Y.A)



blocksymtricirculant(X::HermLaurent, N) = blocksymtricirculant(X.A..., N)
function BlockTridiagonal(X::HermLaurent)
    A,B = X.A
    mortar(Tridiagonal(Fill(Matrix(B'),∞), Fill(A,∞), Fill(B,∞)))
end