using ForwardDiff, LinearAlgebra
import ForwardDiff: jacobian, Dual, gradient, value, partials
import LinearAlgebra: eigvals, eigen

function eigvals(A::Symmetric{<:Dual{Tg,T,N}}) where {Tg,T<:Real,N}
    λ,Q = eigen(Symmetric(value.(parent(A))))
    parts = ntuple(j -> diag(Q' * getindex.(partials.(A), j) * Q), N)
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

eigen(

A = Symmetric([Dual(0.1,(1.0,0.0,0.0)) Dual(0.1,(1.0,0.0,0.0)); 
                Dual(0.0,(0.0,0.0,0.0)) Dual(0.2+2*0.3,(0.0,1.0,2.0))])


(Q'Ȧ*Q - Λ̇) ./ (diag(Λ) .- diag(Λ)')

eigvals(Symmetric([Dual(0.1,(1.0,0.0,0.0)) Dual(0.1,(1.0,0.0,0.0)); 
                Dual(0.0,(0.0,0.0,0.0)) Dual(0.2+2*0.3,(0.0,1.0,2.0))]))



A = c -> Symmetric([c[1] c[1]; 0 c[2]+2c[3]]);
Q = c -> vec(eigen(A(c)).vectors);
jacobian(Q, [0.1,0.2,0.3])
h = 0.00001; (Q([0.1+h,0.2,0.3]) - Q([0.1,0.2,0.3]))/h

λ = c -> maximum(eigvals(A(c)))


h = 0.00001; (λ([0.1+h,0.2,0.3]) - λ([0.1,0.2,0.3]))/h
h = 0.00001; (λ([0.1,0.2+h,0.3]) - λ([0.1,0.2,0.3]))/h

_,Q = eigen(A([0.1,0.2]))
Q[:,1]
[diag(Q'*[1 1; 1 0]*Q) diag(Q'*[0 0; 0 1]*Q)]

Q̇

h = 0.00001; Q̇ = (eigen(A([0.1+h,0.2,0.3])).vectors - eigen(A([0.1,0.2,0.3])).vectors)/h


Q'Ȧ*Q - Λ̇
Q'Q̇*Λ - Q'A(c)*Q̇
Q'Q̇*Λ - Λ*Q'*Q̇




Q̇'Q
Q'Q̇

Q̇'Q + Q'Q̇
Ȧ*Q + A(c)*Q̇  - (Q̇*Λ + Q*Λ̇)
Q'Ȧ*Q + Q'A(c)*Q̇  - (Q'Q̇*Λ + Λ̇)
Q'Ȧ*Q + Q'A(c)*Q̇  - (-Q̇'Q*Λ + Λ̇)
Q'Ȧ*Q + Q'A(c)*Q̇  - (-Q̇'A(c)*Q + Λ̇)
Q'A(c)*Q̇ + Q̇'A(c)*Q - (Λ̇ - Q'Ȧ*Q)
A(c)*Q̇ + Q*Q̇'A(c)*Q - (Q*Λ̇ - Ȧ*Q)
A(c)*Q̇ - Q̇*Q'A(c)*Q - (Q*Λ̇ - Ȧ*Q)
A(c)*Q̇ - Q̇*Λ - (Q*Λ̇ - Ȧ*Q)
(A(c)-Λ[1]*I)*Q̇[:,1]

(A(c)-Λ[1]*I) \ (Q*Λ̇ - Ȧ*Q)[:,1]
Q̇
-Q'A(c)*Q̇

λ̇ = jacobian(λ, [0.1,0.2,0.3])
c = [0.1,0.2,0.3]
Λ̇ = Diagonal(λ̇[:,1])
λ,Q = eigen(A(c))
Λ = Diagonal(λ)
Ȧ = Symmetric([1 1; 0 0])

(A(c) \ (Q*Λ̇ - Ȧ*Q))/2







λ,Q = eigen(Symmetric(getproperty.(parent(A), :value)))





Dual(0.1,(0.1,0.2,0.3))

g = c -> [exp(c[1]+c[2]); sin(c[1]c[2])]
Debugger.@enter jacobian(g,[0.1,0.2])