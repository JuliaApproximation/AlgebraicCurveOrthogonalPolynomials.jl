using BandedMatrices


x₂=-(1/4)*sqrt(4+2*sqrt(2))
x₀=-1/sqrt(2)
dim = 400;
X = BandedMatrix{Float64}(undef, (dim,dim), (2,2))

X[band(0)] = [x₀; 1/2*ones(dim-1)]
X[band(1)] .= X[band(-1)] .= 0;
X[band(2)] = X[band(-2)] = [x₂; 1/4*ones(dim-3)]
X



a₁₂ = (1 + sqrt(2))/4
a₂₁ = (1 - sqrt(2))/4
y₁=(1/4)*sqrt(4-2*sqrt(2))
Y = BandedMatrix{Float64}(undef, (dim,dim), (3,3))
Y[band(0)].=0
Y[band(2)].=Y[band(-2)].=0
d1 = [y₁ repeat([-1/2 a₂₁],1,Int64(round(dim/2)))]
d3 = repeat([0 a₁₂],1,Int64(round(dim/2)))
Y[band(1)] = Y[band(-1)] = d1[1:dim-1]
Y[band(3)] = Y[band(-3)] = d3[1:dim-3]
Y

λ_x,Q = eigen(Matrix(X))

λ_x



Q' * X * Q
Q' * Y * Q

λ,Q = eigen(Matrix(X + im * Y))
Q' * Y * Q
Q' * X * Q




Q'Q

w = abs2.(Q[1,:])

sum(w)

histogram(angle.(λ[2:end]); nbins=50, weights=w[2:end])
