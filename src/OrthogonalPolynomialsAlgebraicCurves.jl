module OrthogonalPolynomialsAlgebraicCurves
using FastGaussQuadrature, FastTransforms, SpecialFunctions, LinearAlgebra, BlockBandedMatrices, BlockArrays, 
    ForwardDiff, OrthogonalPolynomialsQuasi, DomainSets, StaticArrays, ContinuumArrays, QuasiArrays

import ForwardDiff: jacobian
import ForwardDiff: jacobian, Dual, gradient, value, partials
import LinearAlgebra: eigvals, eigen, isapprox, SymTridiagonal, norm, factorize
import FastGaussQuadrature: jacobimoment
import QuasiArrays: DefaultQuasiArrayStyle, cardinality
import Base: in, axes, getindex, broadcasted, tail, +, -, *, /, \, convert, OneTo, show
import ContinuumArrays: Weight, grid, ℵ₁
import OrthogonalPolynomialsQuasi: checkpoints, ShuffledRFFT, TransformFactorization, ldiv, paddeddata

import BlockArrays: block, blockindex, _BlockedUnitRange
import BlockBandedMatrices: BlockTridiagonal

export quarticjacobi, blocksymtricirculant, unroll, randspeccurve, speccurve, specgrid, speccurvemat, symroll, symunroll, spec2alg,
        wedgep, wedgeq, wedger, wedgetransform, plan_wedgetransform, plan_squaretransform, gausswedge, JacobiWedge, LegendreSquare, gausssquare,
        HermLaurent, jointeigen, jointeigvals, BlockTridiagonal, LegendreCircle, UltrasphericalCircle, Block, SVector, CircleCoordinate, UltrasphericalArc

function eigvals(A::Symmetric{<:Dual{Tg,T,N}}) where {Tg,T<:Real,N}
    λ,Q = eigen(Symmetric(value.(parent(A))))
    parts = ntuple(j -> diag(Q' * getindex.(partials.(A), j) * Q), N)
    Dual{Tg}.(λ, tuple.(parts...))
end

function eigvals(A::Hermitian{<:Complex{<:Dual{Tg,T,N}}}) where {Tg,T<:Real,N}
    λ,Q = eigen(Hermitian(value.(real.(parent(A))) .+ im .* value.(imag.(parent(A)))))
    parts = ntuple(j -> diag(real.(Q' * (getindex.(partials.(real.(A)) .+ im .* partials.(imag.(A)), j)) * Q)), N)
    Dual{Tg}.(λ, tuple.(parts...))
end

# A ./ (λ - λ') but with diag special cased
function _lyap_div!(A, λ)
    for (j,μ) in enumerate(λ), (k,λ) in enumerate(λ)
        if k ≠ j
            A[k,j] /= μ - λ
        end
    end
    A
end

function eigen(A::Symmetric{<:Dual{Tg,T,N}}) where {Tg,T<:Real,N}
    λ = eigvals(A)
    _,Q = eigen(Symmetric(value.(parent(A))))
    parts = ntuple(j -> Q*_lyap_div!(Q' * getindex.(partials.(A), j) * Q - Diagonal(getindex.(partials.(λ), j)), value.(λ)), N)
    Eigen(λ,Dual{Tg}.(Q, tuple.(parts...)))
end



"""
    blocksymtricirculant(A,B, N)

Creates a Block N x N  Symmetric-Tridiagonal-Circulant matrix with diagonal
blocks A and off-diagonal block B.
"""
function blocksymtricirculant(A, B, N)
    M = size(A,1)
    ret = BlockMatrix(zeros(eltype(A),M*N,M*N), Fill(M,N), Fill(M,N))
    for K = 1:N 
        ret[Block(K,K)] = A 
    end
    for K = 1:N-1 
        ret[Block(K,K+1)] = B
        ret[Block(K+1,K)] = B' 
    end
    ret[Block(1,N)] = B'
    ret[Block(N,1)] = B
    ret
end

function symunroll(a)
    m = length(a)
    n = (isqrt(8m+1)-1) ÷ 2
    @assert sum(1:n) == m # double check formula...
    A = similar(a, n, n)
    k̃ = 1
    for j = 1:n, k = 1:j
        A[k,j] = A[j,k] = a[k̃]
        k̃ += 1
    end
    A
end

symroll(A) = mortar([A[1:k,k] for k=1:size(A,2)])

# a holds the non-symmetric entries of A
function _unroll(a, b)
    n = isqrt(length(b))
    B = reshape(b,n,n)
    @assert length(a) == sum(1:n)
    A = similar(a, n, n)
    k̃ = 1
    for j = 1:n, k = 1:j
        A[k,j] = A[j,k] = a[k̃]
        k̃ += 1
    end
    A,B
end

# split into a and b terms
function _unroll(x)
    N = length(x)
    n = (isqrt(1 + 24N)-1) ÷ 6
    m = sum(1:n)
    _unroll(x[1:m], x[m+1:end])
end

"""
unroll variables into Symmetric Ax and non-symmetric Bx
"""
function unroll(x)
    @assert iseven(length(x))
    N = length(x) ÷ 2
    Ax,Bx = _unroll(x[1:N])
    Ay,By = _unroll(x[N+1:end])
    Ax,Bx,Ay,By
end

abstract type AlgebraicOrthogonalPolynomial{d,T} <: Basis{T} end

include("circle.jl")
include("arc.jl")
include("wedge.jl")
include("square.jl")

include("quartic.jl")

include("algcurvapprox.jl")

include("hermlaurent.jl")

end # module
