using ApproxFun, Test
import ApproxFun: ArraySpace


# this finds the OPs and recurrence for matrix weights. We think in terms of vector functions
# f(x) = Σ P_k(x) f_k = [P_0 P_1 …]*[f_0,f_1,…]
# so that the inner product is
# <f,g> = f'W*g
# means we want
# <P_k,P_k> = P_k'W*P_k = I_k 
# and zero for j ≠ k
# and 
# f_k = <P_k,f>
#
# The columns of a degree m matrix polynomial F
# are of course degree m vector polynomials f and
# can be reexpanded as above. Thus we can expand
# a degree m matrix polynomial as
# 
# F(x) = Σ P_k(x) F_k
# where
# F_k = <P_k,F>
# 
# Thus we can reexpand 
#
# x*P_k = P_{k+1} C_k + P_k A_k + P_{k-1} B_{k-1}
#
# so that x*f(x) = [P_0 P_1 …] * [A_0 B_0 
#                                 C_0 A_1 B_1 …] * [f_0,f_1,…]



function lanczos(w::Fun{<:ArraySpace},N)
    x = Fun(identity,domain(w))

    ip = (f,g) -> sum(transpose(f)*w*g)

    f1=Fun(inv(sqrt(sum(w))),ArraySpace(space(x),size(w)...))

    P = Array{Fun}(undef, N + 1)
    β = Array{eltype(w)}(undef, N)
    γ = Array{eltype(w)}(undef, N)

    P[1] = f1

    v = x*P[1]
    β[1] = ip(P[1],v)

    v = v - P[1]*β[1]
    γ[1] = real(sqrt(ip(v,v)))

    P[2] = v*inv(γ[1])

    for k = 2:N
        v = x*P[k] - P[k-1]*γ[k-1]
        β[k] = ip(P[k],v)
        v = v - P[k]*β[k]
        γ[k] = real(sqrt(ip(v,v)))
        P[k+1] = v*inv(γ[k])
    end

    P,β,γ
end

Q = qr(randn(2,2)).Q
w = Fun(Q*Diagonal([1,2])*Q', ArraySpace(Chebyshev(),2,2))

w = Fun(x -> [2+x 1; 1 x^2])
P,β,γ = lanczos(w,10)

w = Fun(x -> [2 x; x x^2])
P,β,γ = lanczos(w,100)


x = 0.1
k = 5; 
@test x*P[k](x) ≈ P[k-1](x)*γ[k-1]' + P[k](x)*β[k] + P[k+1](x)*γ[k]



w = Fun(x -> [1 x; x x^2]/sqrt(1-x^2), ArraySpace(JacobiWeight(-0.5,-0.5,Chebyshev()),2,2))

w(-0.1) |> eigvals

lanczos(w,10)