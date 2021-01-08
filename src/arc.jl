struct Arc{T} <: Domain{T} end
Arc() = Arc{Float64}()
cardinality(::Arc) = ℵ₁

show(io::IO, a::Arc) = print(io, "Arc()")

checkpoints(::Arc{T}) where T = [CircleCoordinate(0.1),CircleCoordinate(1.32),CircleCoordinate(2.9)]

function in(xy::StaticVector{2}, d::Arc)
    x,y = xy
    norm(xy) == 1 && y ≥ 0
end

const ArcInclusion{T} = Inclusion{CircleCoordinate{T},Arc{T}}
ArcInclusion{T}() where T = Inclusion{CircleCoordinate{T},Arc{T}}(Arc{T}())


"""
y^a on half circle
"""
struct UltrasphericalArcWeight{T} <: Weight{T}
    a::T
end

axes(P::UltrasphericalArcWeight{T}) where T = (ArcInclusion{T}(),)
getindex(P::UltrasphericalArcWeight, xy::StaticVector{2}) = xy[2]^P.a

"""
Ortogonal polynomials w.r.t. y^a for y^2 + x^2 = 1, y ≥ 0.
"""
struct UltrasphericalArc{V,TT,UU} <: AlgebraicOrthogonalPolynomial{2,V}
    a::V
    T::TT
    U::UU
end
UltrasphericalArc{V}(a, T::TT, U::UU) where {V,TT,UU} = UltrasphericalArc{V,TT,UU}(a,T,U)

function UltrasphericalArc{V}(a) where V
    T = SemiclassicalJacobi(2, -1/2+a/2, 0, -1/2+a/2)
    U = SemiclassicalJacobi(2,  1/2+a/2, 0, 1/2+a/2, T)
    UltrasphericalArc{V}(a, Normalized(T), Normalized(U))
end

function UltrasphericalArc{V}(a, P) where V
    T = SemiclassicalJacobi(2, -1/2+a/2, 0, -1/2+a/2, P.T.P)
    U = SemiclassicalJacobi(2,  1/2+a/2, 0, 1/2+a/2, T)
    UltrasphericalArc{V}(a, Normalized(T), Normalized(U))
end


UltrasphericalArc(a::T) where T = UltrasphericalArc{float(T)}(a)
UltrasphericalArc(a::T, P) where T = UltrasphericalArc{float(T)}(a, P)
UltrasphericalArc() = UltrasphericalArc(0)

axes(P::UltrasphericalArc{T}) where T = (ArcInclusion{T}(), _BlockedUnitRange(1:2:∞))

==(P::UltrasphericalArc, Q::UltrasphericalArc) = P.a == Q.a

function getindex(P::UltrasphericalArc{T}, xy::StaticVector{2}, j::BlockIndex{1}) where T
    x,y = xy
    K,k = block(j),blockindex(j)
    K == Block(1) && return P.T[1-y,1]
    k == 1 ? x*P.U[1-y,Int(K)-1] :  P.T[1-y,Int(K)] 
end

function getindex(P::UltrasphericalArc, xy::StaticVector{2}, j::Int)
    j == 1 && return P[xy, Block(1)[1]]
    P[xy, Block((j ÷ 2)+1)[1+isodd(j)]]
end

summary(io::IO, P::UltrasphericalArc{V}) where V = print(io, "UltrasphericalArc($(P.a))")

function ldiv(Pn::SubQuasiArray{T,2,<:UltrasphericalArc,<:Tuple{Inclusion,OneTo}}, f::AbstractQuasiVector{V}) where {T,V}
    _,jr = parentindices(Pn)
    P = parent(Pn)
    N = maximum(jr)
    ret = Array{T}(undef, N)
    ret[1:2:end] = P.T[:,1:(N+1) ÷ 2] \ BroadcastQuasiVector{Float64}(y -> f[CircleCoordinate(asin(1-y))] + f[CircleCoordinate(π-asin(1-y))], axes(P.T,1))
    if N > 1
        ret[2:2:end] = P.U[:,1:N÷2] \ BroadcastQuasiVector{Float64}(function(y)
            xy = CircleCoordinate(asin(1-y))
            xỹ = CircleCoordinate(π-asin(1-y))
            (f[xy] - f[xỹ])/xy[1]
        end, axes(P.T,1))
    end
    ldiv!(2,ret)
end

abstract type AbstractUltrasphericalArcJacobi{T} <: AbstractBlockBandedMatrix{T} end

ArrayLayouts.MemoryLayout(::Type{<:AbstractUltrasphericalArcJacobi}) = LazyBandedMatrices.LazyBandedBlockBandedLayout()
Base.BroadcastStyle(::Type{<:AbstractUltrasphericalArcJacobi}) = LazyBandedMatrices.LazyArrayStyle{2}()

struct UltrasphericalArcJacobiX{T} <: AbstractUltrasphericalArcJacobi{T}
    R
end

UltrasphericalArcJacobiX(R) = UltrasphericalArcJacobiX{eltype(R)}(R)

struct UltrasphericalArcJacobiY{T} <: AbstractUltrasphericalArcJacobi{T}
    X_T
    X_U
end

UltrasphericalArcJacobiY(X_T, X_U) = UltrasphericalArcJacobiY{promote_type(eltype(X_T),eltype(X_U))}(X_T,X_U)



function BlockArrays.viewblock(X::UltrasphericalArcJacobiX{T}, kj::Block{2}) where T
    k,j = kj.n
    R = X.R
    k == j == 1 && return zeros(T,1,1)
    (k,j) == (2,1) && return reshape([R[1,1]; zero(T)], 2, 1)
    (k,j) == (1,2) && return [R[1,1] zero(T)]
    k == 1 && return zeros(T,1,2)
    j == 1 && return zeros(T,2,1)
    k == j && return [zero(T) R[j-1,j]; R[j-1,j] zero(T)]
    k == j+1 && return [zero(T) R[j,j]; R[j-1,j+1] zero(T)]
    k+1 == j && return [zero(T) R[k-1,k+1]; R[k,k] zero(T)]
    zeros(T,2,2)
end

function BlockArrays.viewblock(Y::UltrasphericalArcJacobiY{T}, kj::Block{2}) where T
    k,j = kj.n
    X_T, X_U = Y.X_T, Y.X_U
    k == j == 1 && return reshape([1-X_T[1,1]],1,1)
    (k,j) == (2,1) && return reshape([zero(T),-X_T[2,1]], 2, 1)
    (k,j) == (1,2) && return [zero(T) -X_T[1,2]]
    k == 1 && return zeros(T,1,2)
    j == 1 && return zeros(T,2,1)
    k == j && return [1-X_U[k-1,j-1] zero(T); zero(T) 1-X_T[k,j]]
    return [-X_U[k-1,j-1] zero(T); zero(T) -X_T[k,j]]
end

function getindex(X::AbstractUltrasphericalArcJacobi, k::Int, j::Int)
    ki,ji = findblockindex.(axes(X), (k,j))
    X[block(ki),block(ji)][blockindex(ki),blockindex(ji)]
end

axes(::AbstractUltrasphericalArcJacobi) = (_BlockedUnitRange(1:2:∞),_BlockedUnitRange(1:2:∞))

blockbandwidths(::AbstractUltrasphericalArcJacobi) = (1,1)
subblockbandwidths(::AbstractUltrasphericalArcJacobi) = (1,1)

function jacobimatrix(::Val{1}, P::UltrasphericalArc)
    R = P.U \ P.T;
    UltrasphericalArcJacobiX(R)
end

function jacobimatrix(::Val{2}, P::UltrasphericalArc)
    X_T = jacobimatrix(P.T);
    X_U = jacobimatrix(P.U);
    UltrasphericalArcJacobiY(X_T, X_U)
end

struct UltrasphericalArcConversion{T} <: AbstractBlockBandedMatrix{T}
    R_T
    R_U
end

axes(::UltrasphericalArcConversion) = (_BlockedUnitRange(1:2:∞),_BlockedUnitRange(1:2:∞))

function UltrasphericalArcConversion(P, Q)
    @assert Q.a == P.a + 2
    R_T = Q.T \ P.T
    R_U = Q.U \ P.U
    UltrasphericalArcConversion(R_T, R_U)
end

function BlockArrays.getblock(R::UltrasphericalArcConversion{T}, k::Int, j::Int) where T
    R_T, R_U = R.R_T, R.R_U
    k == j == 1 && return reshape([R_T[1,1]],1,1)
    j == 1 && return reshape([zero(T),R_T[k,1]], 2, 1)
    k == 1 && return [zero(T) R_T[1,j]]
    k == j && return [R_U[k-1,j-1] zero(T); zero(T) R_T[k,j]]
    return [R_U[k-1,j-1] zero(T); zero(T) R_T[k,j]]
end

function getindex(X::UltrasphericalArcConversion, k::Int, j::Int)
    ki,ji = findblockindex.(axes(X), (k,j))
    X[block(ki),block(ji)][blockindex(ki),blockindex(ji)]
end




BlockBandedMatrices.MemoryLayout(::Type{<:AbstractUltrasphericalArcJacobi}) = BlockBandedMatrices.BlockBandedLayout()
BlockBandedMatrices.MemoryLayout(::Type{<:UltrasphericalArcConversion}) = BlockBandedMatrices.BlockBandedLayout()

function \(Q::UltrasphericalArc{T}, P::UltrasphericalArc{V}) where {T,V}
    P == Q && return FillArrays.SquareEye{promote_type(T,V)}((axes(P,2),))
    UltrasphericalArcConversion(P, Q)
end
    