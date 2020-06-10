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


####
# comjac
####
using BlockArrays
import OrthogonalPolynomialsAlgebraicCurves: cmjac, symjac, trjac

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

"""
 Squares B/z + A + B'*z 
"""
lrntsquare(B, A) = (B^2, B*A+A*B, B*B'+A^2 + B'B)
"""
 Squares C/z^2 + B/z + A + B'*z + C'*z^2
"""
lrntsquare(C, B, A) = (C^2, C*B + B*C, C*A + B^2 + A*C, C*B' + B*A + A*B + B'*C, C*C' + B*B' + A^2 + B'B + C'C)


function newunroll(pin)
    p,c = pin[1:end-3],pin[end-2:end]
    Aˣ, Bˣ = comunroll(p)
    newunroll(Aˣ, Bˣ, c)
end
function newunroll(Aˣ, Bˣ, c)
    N = size(Aˣ,1)

    S = symjac(N)
    Tr = trjac(N)
    J = [0S             cmjac(Bˣ);
        cmjac(Bˣ)*S    cmjac(Aˣ);
        cmjac(Aˣ)*S   (cmjac(Bˣ)*Tr + cmjac(Bˣ'))]

    K = qr_nullspace(3,J)
    K = K*(K')[:,1:3] # normalise to enforce continuity

    # K = nullspace(J)

    (Aʸ, Bʸ) = comunroll(K*c)
    (Aˣ, Bˣ),(Aʸ, Bʸ)
end


using BandedMatrices
N = 4
Aˣ = zeros(N,N)
Bˣ = zeros(N,N)
Bˣ[band(1)] .= 0.5
Bˣ[band(-1)] .= -0.5
Bˣ[band(3)] .= -0.5
Bˣ[band(-3)] .= 0.5

(Aˣ, Bˣ),(Aʸ, Bʸ) = newunroll(Aˣ, Bˣ, randn(10))
scatter(vec(specgrid(X,Y)))

# circle
function nl(p)
    (Aˣ, Bˣ),(Aʸ, Bʸ) = newunroll(p)
    X² = lrntsquare(Bˣ, Aˣ); # X⁴ = lrntsquare(X²...)
    # X² = tuple(zeros(4,4),zeros(4,4),X²...)
    Y² = lrntsquare(Bʸ, Aʸ); # Y⁴ = lrntsquare(Y²...)
    # Y² = tuple(zeros(4,4),zeros(4,4),Y²...)

    vcat(vec.(X² .+ Y² .- (0I,0I,I))...,
        (eigvals(Symmetric(Aˣ - Bˣ - Bˣ')) .- (-1,-1))...,
        (eigvals(Symmetric(Aʸ - Bʸ - Bʸ')))...,
        (eigvals(Symmetric(Aˣ + Bˣ + Bˣ')) .- (1,1))...,
        Aʸ[1,2]
        )
end

N = 2
p = randn(sum(1:N)+N^2+3)
p = p -  jacobian(nl,p) \ nl(p); norm(nl(p))
(Aˣ, Bˣ),(Aʸ, Bʸ) = newunroll(p)
scatter(vec(specgrid(X,Y)))




function nl(p)
    (Aˣ, Bˣ),(Aʸ, Bʸ) = newunroll(p)
    X² = lrntsquare(Bˣ, Aˣ); X⁴ = lrntsquare(X²...)
    # X² = tuple(zeros(4,4),zeros(4,4),X²...)
    Y² = lrntsquare(Bʸ, Aʸ); Y⁴ = lrntsquare(Y²...)
    # Y² = tuple(zeros(4,4),zeros(4,4),Y²...)

    vcat(vec.(X⁴ .+ Y⁴ .- (0I,0I,0I,0I,I))...,
        (eigvals(Symmetric(Aˣ - Bˣ - Bˣ')) .- (-1,-1,-1,-1))...,
        (eigvals(Symmetric(Aˣ + Bˣ + Bˣ')) .- (1,1,1,1))...,
        Aˣ[1,2],
        Aˣ[1:2,3],
        Aˣ[1:3,4]
        )
end

N = 4
p = randn(sum(1:N)+N^2+3)
p = p -  jacobian(nl,p) \ nl(p); norm(nl(p))

(Aˣ, Bˣ),(Aʸ, Bʸ) = newunroll(p)

scatter(vec(specgrid(X,Y)))

pin = p

nl(p̃)-nl(p)
jacobian(nl,p)
h = 0.0001; p̃ = p .+ [h; zeros(length(p)-1)]; (nl(p̃) - nl(p))/h

Q1 = nl(p)
Q2 = nl(p̃)
(Q1 - Q2*Q2'*Q1)




cmjac(Bˣ) * symjac(N)

scatter(vec(specgrid(X,Y)))