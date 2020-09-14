function gausswedge(n)
    x,w = gaussradau(n)
    reverse!(x); reverse!(w)
    x .= (1 .- x)./2; ldiv!(2, w);
    [x; ones(length(x)-1)], [ones(length(x)); reverse!(x[1:end-1])], [w[1:end-1]; 2w[end]; reverse!(w[1:end-1])]
end

function wedgep(n, x, y)
    P = legendre(0..1)
    P[x,n+1] + P[y,n+1] - 1
end

function wedgeq(n, x, y)
    jacobi(2,0,0..1)
    sqrt(n) * ((1-x) * P²⁰[x,n] - (1-y)*P²⁰[y,n])
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