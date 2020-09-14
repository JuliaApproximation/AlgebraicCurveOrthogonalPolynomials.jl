function gausswedge(n)
    x,w = gaussradau(n)
    reverse!(x); reverse!(w)
    x .= (1 .- x)./2; ldiv!(2, w);
    [x; ones(length(x)-1)], [ones(length(x)); reverse!(x[1:end-1])], [w[1:end-1]; 2w[end]; reverse!(w[1:end-1])]
end

function wedgep(n, a, b, c, x, y)
    Pᶜᵃ = jacobi(c,a,0..1)
    Pᶜᵇ = jacobi(c,b,0..1)
    Pᶜᵃ[x,n+1] + Pᶜᵇ[y,n+1] - binomial(n+c,n)
end

function wedgeq(n, a, b, c, x, y)
    Pᶜᵃ = jacobi(c+2,a,0..1)
    Pᶜᵇ = jacobi(c+2,b,0..1)
    pochhammer(c+a+2,n)/pochhammer(a+1,n-1)*(1-x) * Pᶜᵃ[x,n] - pochhammer(c+b+2,n)/pochhammer(b+1,n-1) * (1-y)*Pᶜᵇ[y,n]
end

function wedgemassmatrix(n)
    x,y,w = gausswedge(n)
    N = 2n-1
    ret = Vector{Float64}(undef, N)
    ret[1] = dot(p0, Diagonal(w), p0)
    for m = 1:n-1
        p0 = wedgep.(m, x, y)
        q0 = wedgeq.(m, x, y)
        ret[2m] = dot(p0, Diagonal(w), p0)
        ret[2m+1] = dot(q0, Diagonal(w), q0)
    end
    PseudoBlockArray(Diagonal(ret), [1; fill(2,n-1)], [1; fill(2,n-1)])
end

function plan_wedgetransform(n)
    N = 2n-1
    x,y,w = gausswedge(n)
    ret = Array{Float64}(undef, N, N)
    p0 = wedgep.(0, x, y)
    σ = dot(p0, Diagonal(w), p0)
    ret[1,:] .= p.(0, x, y) .* w ./ σ
    for m = 1:n-1
        p0 = wedgep.(m, x, y)
        q0 = wedgeq.(m, x, y)
        σ = dot(p0, Diagonal(w), p0)
        μ = dot(q0, Diagonal(w), q0)
        ret[2m,:] .= wedgep.(m, x, y) .* w ./ σ
        ret[2m+1,:] .= wedgeq.(m, x, y) .* w ./ μ
    end
    PseudoBlockArray(ret, [1; fill(2,n-1)], [N])
end

wedgetransform(v::AbstractVector) = plan_wedgetransform(length(v) ÷ 2 + 1) * v



struct Wedge{T} <: EuclideanDomain{2,T} end
Wedge() = Wedge{Float64}()

function in(p::SVector{2}, d::Wedge)
    x,y = p
    (x == 1 && 0 ≤ y ≤ 1) || (y == 1 && 0 ≤ x ≤ 1)
end

struct WedgeLegendre{T} <: OrthogonalPolynomial{T} end
WedgeLegendre() = WedgeLegendre{Float64}()

axes(P::WedgeLegendre) = (Inclusion(Wedge()),blockedrange([1; Fill(2,∞)]))

function getindex(P::WedgeLegendre, xy::SVector{2}, j::BlockIndex{1})
    K,k = block(j),blockindex(j)
    k == 1 ? wedgep(Int(K)-1, xy...) : wedgeq(Int(K)-1, xy...)
end

function getindex(P::WedgeLegendre, xy::SVector{2}, j::Int)
    j == 1 && return P[xy, Block(1)[1]]
    P[xy, Block((j ÷ 2)+1)[1+isodd(j)]]
end