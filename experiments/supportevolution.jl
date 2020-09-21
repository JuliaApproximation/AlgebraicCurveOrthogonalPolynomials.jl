using Plots

B = [-0.5 0 0 0; 0 0 0 1; 0 0 -0.5 0; 0 0 0 0]
A = randn(4,4); A = A + A'







θ = range(0,π; length=200); plot(eigvals.(Xε[exp.(im*θ)]))

X = HermLaurent(0A,B)
λ,Q = eigen(X[1])
ε = 0.0001;
f = function(ε)
    Xε = HermLaurent(ε*A,B)
    eigvals(Xε[1])
end

Q'X[1]Q
Q[:,1:3]'*A*Q[:,1:3] |> eigvals
Q[:,4]'*A*Q[:,4]
f(0.0001)
f(0.001)
ForwardDiff.derivative(f,0.0)

ε  = 0.00001; (f(ε)-f(0))/ε



diag(Q'*A*Q)



Q'*X[1]*Q

eigvals(X[1])


minimum(map(minimum,eigvals.(Xε[exp.(im*θ)])))
maximum(map(maximum,eigvals.(Xε[exp.(im*θ)])))



