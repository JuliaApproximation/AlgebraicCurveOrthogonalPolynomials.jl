"""
CubicCurve represents y^2 = x * (1-x) * (t-x) for x in 0..1
"""
struct CubicCurve{T} <: Domain{T}
    t::T
end

CubicCurve{T}() where T = CubicCurve{T}(zero(T))
CubicCurve() = CubicCurve{Float64}()
CubicCurve(a::CubicCurve) = a
CubicCurve{T}(a::CubicCurve) where T = CubicCurve{T}(a.h)
cardinality(::CubicCurve) = ℵ₁

show(io::IO, a::CubicCurve) = print(io, "CubicCurve($(a.t))")

# TODO: Make sure in domain
checkpoints(d::CubicCurve{T}) where T = [SVector(0.1, sqrt(0.1*(1-0.1)*(d.t-0.1)))]

function in(xy::StaticVector{2}, d::CubicCurve)
    x,y = xy
    0 ≤ x ≤ 1 && y^2 ≈ x*(1-x)*(d.t-x)
end

const CubicCurveInclusion{T} = Inclusion{SVector{2,T},CubicCurve{T}}
CubicCurveInclusion{T}() where T = Inclusion{SVector{2,T},CubicCurve{T}}(CubicCurve{T}())
CubicCurveInclusion{T}(a::Number) where T = Inclusion{SVector{2,T},CubicCurve{T}}(CubicCurve{T}(a))
CubicCurveInclusion(a) = CubicCurveInclusion{typeof(a)}(a)



struct LegendreCubic{V,PP,QQ} <: AlgebraicOrthogonalPolynomial{2,V}
    t::V
    P::PP
    Q::QQ
end
LegendreCubic{V}(t, P::PP, Q::QQ) where {V,PP,QQ} = LegendreCubic{V,PP,QQ}(t,P,Q)

function LegendreCubic{V}(t) where V
    P = SemiclassicalJacobi(t, 0, 0, 0)
    Q = SemiclassicalJacobi(t,  1, 1, 1, P)
    LegendreCubic{V}(t, Normalized(P), Normalized(Q))
end


LegendreCubic(t) = LegendreCubic{float(typeof(t))}(t)

# block sizes are [1,2,3,3,3,3…] so cum is [1,3,6,9,…]
axes(P::LegendreCubic{T}) where T = (CubicCurveInclusion(P.t), _BlockedUnitRange(Vcat(1, 3:3:∞)))

==(P::LegendreCubic, Q::LegendreCubic) = P.t == Q.t

function getindex(P::LegendreCubic{T}, xy::StaticVector{2}, J::Block{1}) where T
    x,y = xy
    t = P.t
    J == Block(1) && return [one(T)]
    J == Block(2) && return [P.P[x,2], y*P.Q[x,1]]
    J̃ = Int(J)
    m = J̃ ÷ 2
    isodd(J̃) && return [P.P[x,3m+1], y*P.Q[x, 3m-1], P.P[x, 3m]]
    #iseven(J̃)
    return [y*P.Q[x, 3m-2], P.P[x,3m-1], y*P.Q[x, 3m-3]]
end

function getindex(P::LegendreCubic{T}, xy::StaticVector{2}, Jj::BlockIndex{1}) where T
    x,y = xy
    J,j = block(Jj),blockindex(Jj)
    P[xy,J][j]
end

getindex(P::LegendreCubic{T}, xy::StaticVector{2}, Jj::BlockOneTo) where T = layout_getindex(P, xy, Jj)

function getindex(P::LegendreCubic, xy::StaticVector{2}, j::Int)
    j == 1 && return P[xy, Block(1)[1]]
    P[xy, Block((j ÷ 2)+1)[1+isodd(j)]]
end

summary(io::IO, P::LegendreCubic{V}) where V = print(io, "LegendreCubic($(P.t))")
