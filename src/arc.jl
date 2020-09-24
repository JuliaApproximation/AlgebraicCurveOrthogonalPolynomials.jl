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
Ortogonal polynomials w.r.t. y^a
"""
struct UltrasphericalArc{V,TT,UU} <: AlgebraicOrthogonalPolynomial{2,V}
    a::V
    T::TT
    U::UU
end
UltrasphericalArc{V}(a, T::TT, U::UU) where {V,TT,UU} = UltrasphericalArc{V,TT,UU}(a,T,U)

function UltrasphericalArc{V}(a) where V 
    P₊ = jacobi(1/2,0,0..1)
    y = axes(P₊,1)
    x = @. sqrt(1 - y^2)
    U = LanczosPolynomial(y .^ a .* x, P₊)
    P₋ = jacobi(-1/2,0,0..1)
    T = LanczosPolynomial(y .^ a ./ x, P₋)
    UltrasphericalArc{V}(a, T, U)
end

UltrasphericalArc(a::T) where T = UltrasphericalArc{float(T)}(a)
UltrasphericalArc() = UltrasphericalArc(0)

axes(P::UltrasphericalArc{T}) where T = (ArcInclusion{T}(), _BlockedUnitRange(1:2:∞))

function getindex(P::UltrasphericalArc{T}, xy::StaticVector{2}, j::BlockIndex{1}) where T
    x,y = xy
    K,k = block(j),blockindex(j)
    K == Block(1) && return P.T[y,1]
    k == 1 ? x*P.U[y,Int(K)-1] :  P.T[y,Int(K)] 
end

function getindex(P::UltrasphericalArc, xy::StaticVector{2}, j::Int)
    j == 1 && return P[xy, Block(1)[1]]
    P[xy, Block((j ÷ 2)+1)[1+isodd(j)]]
end

show(io::IO, P::UltrasphericalArc{V}) where V = print(io, "UltrasphericalArc($(P.a))")
show(io::IO, ::MIME"text/plain", P::UltrasphericalArc) = show(io, P)

function ldiv(Pn::SubQuasiArray{T,2,<:UltrasphericalArc,<:Tuple{Inclusion,OneTo}}, f::AbstractQuasiVector{V}) where {T,V}
    _,jr = parentindices(Pn)
    P = parent(Pn)
    N = maximum(jr)
    ret = Array{T}(undef, N)
    ret[1:2:end] = P.T[:,1:(N+1) ÷ 2] \ BroadcastQuasiVector{Float64}(y -> f[CircleCoordinate(asin(y))] + f[CircleCoordinate(π-asin(y))], axes(P.T,1))
    if N > 1
        ret[2:2:end] = P.U[:,1:N÷2] \ BroadcastQuasiVector{Float64}(function(y)
            xy = CircleCoordinate(asin(y))
            xỹ = CircleCoordinate(π-asin(y))
            (f[xy] - f[xỹ])/xy[1]
        end, axes(P.T,1))
    end
    ldiv!(2,ret)
end