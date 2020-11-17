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
    T = SemiclassicalJacobi(2, -1/2-a/2, 0, -1/2-a/2)
    U = SemiclassicalJacobi(2,  1/2-a/2, 0, 1/2-a/2, T)
    UltrasphericalArc{V}(a, Normalized(T), Normalized(U))
end

UltrasphericalArc(a::T) where T = UltrasphericalArc{float(T)}(a)
UltrasphericalArc() = UltrasphericalArc(0)

axes(P::UltrasphericalArc{T}) where T = (ArcInclusion{T}(), _BlockedUnitRange(1:2:∞))

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

struct UltrasphericalArcJacobiX{T} <: AbstractBlockBandedMatrix{T}
    R
end

UltrasphericalArcJacobiX(R) = UltrasphericalArcJacobiX{eltype(R)}(R)

function BlockArrays.getblock(X::UltrasphericalArcJacobiX{T}, k::Int, j::Int) where T
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

function getindex(X::UltrasphericalArcJacobiX, k::Int, j::Int)
    ki,ji = findblockindex.(axes(X), (k,j))
    X[block(ki),block(ji)][blockindex(ki),blockindex(ji)]
end

axes(::UltrasphericalArcJacobiX) = (_BlockedUnitRange(1:2:∞),_BlockedUnitRange(1:2:∞))

blockbandwidths(::UltrasphericalArcJacobiX) = (1,1)
subblockbandwidths(::UltrasphericalArcJacobiX) = (1,1)

function jacobimatrix(::Val{1}, P::UltrasphericalArc)
    R = P.U \ P.T;
    UltrasphericalArcJacobiX(R)
end

BlockBandedMatrices.MemoryLayout(::Type{<:UltrasphericalArcJacobiX}) = BlockBandedMatrices.BlockBandedLayout()