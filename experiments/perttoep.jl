using OrthogonalPolynomialsAlgebraicCurves, Plots, ForwardDiff
import OrthogonalPolynomialsAlgebraicCurves: cm, jacobian


###
# I give up on fitting algebraic curves!!!
# 
# Now let's just perturb the 4 x 4 symbols
#
# Start with two circles
###

λˣ, V = eigen([0 -0.5 0 0;
 0.5 0 0 0;
 0 0 0 -0.5;
 0 0 0.5 0]) # .+ 0.0001 .* randn.())
V*Diagonal(λˣ)*inv(V)

c = K \ [fill(0.5,4)...; symroll(zeros(4,4))...]

OrthogonalPolynomialsAlgebraicCurves.comunroll(V,K*c)[1]

(Aˣ,Bˣ),(Aʸ,Bʸ) = (t -> real.(t)).(speccurvemat(Symmetric(zeros(4,4)), (λˣ, V), c .+ randn.()))

X = z -> Aˣ + Bˣ/z + z*Bˣ'
Y = z -> Aʸ + Bʸ/z + z*Bʸ'

spec2alg(X,Y)

scatter(vec(specgrid(X,Y)))



###
# new param
###

Aˣ,Bˣ = zeros(N,N),Matrix(0.5I,4,4)
N = 4
p = randn(sum(1:N) + N^2)

function comunroll(p) 
    m = length(p)
    N = round(Int,(-1 + sqrt(1 + 24m))/6)
    symunroll(p[1:sum(1:N)]),reshape(p[sum(1:N)+1:end],N,N)
end

conds = function(p)
    Aʸ, Bʸ = comunroll(p)
    [vec(cm(Bˣ,Bʸ));
        vec(cm(Aˣ,Bʸ) + cm(Bˣ,Aʸ));
        vec(cm(Bˣ,Bʸ') + cm(Bˣ',Bʸ) + cm(Aˣ, Aʸ))]
end

K = nullspace(jacobian(conds,p))

Aʸ,Bʸ = comunroll(K*randn(size(K,2)))

z = exp(0.1im)
cm(X(z),Y(z))

scatter(vec(specgrid(X,Y)))

## no Aʸ

conds = function(p)
    Bʸ = reshape(p,N,N)
    [vec(cm(Bˣ,Bʸ));
    vec(cm(Bˣ,Bʸ') + cm(Bˣ',Bʸ) )]
end


K = nullspace(jacobian(conds,randn(N^2)))

Aʸ = zeros(N,N)


Bʸ = reshape(K*randn(size(K,2)),N,N)
Bʸ = [0 -0.5 0 0;
        0.5 0 0 0;
        0 0 0 -0.5;
        0 0 0.5 0]

C = randn(N,N); Bʸ = C - C'

Bʸ .+= randn.()

X = z -> Aˣ + Bˣ/z + z*Bˣ'
Y = z -> Aʸ + Bʸ/z + z*Bʸ'
scatter(vec(specgrid(X,Y)))

Aˣ = zeros(N,N)
Aʸ
Bˣ = randn(N,N)
K = nullspace(jacobian(conds,randn(N^2)))
Bʸ = reshape(K*randn(2),N,N)
scatter(vec(specgrid(X,Y)))

spec2alg(X,Y)