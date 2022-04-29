using AlgebraicCurveOrthogonalPolynomials, ClassicalOrthogonalPolynomials, FillArrays, Test, Plots

##
# Let's start with the circle, parameterised by symbols:
#
#   X(z) = Eye(2) * (z/2 + 1/2z)
#   Y(z) = [0 1; -1 0]*z/2 + [0 -1; 1 0]/2z
##

X = hermlaurent(Zeros(2,2), Eye(2)/2) # z -> Eye(2) * (z/2 + 1/2z)
Y = hermlaurent(Zeros(2,2), [0 -1/2; 1/2 0]) #z -> [0 z/2-1/2z; 1/2z-z/2 0]

# The tangent is defined by taking derivatives. But it will be derivatives w.r.t.
# θ where z = exp(im*θ).

Ẋ = diff(X) # z -> Eye(2) * im*(z/2 - 1/2z)
Ẏ = diff(Y) # z -> [0 im*(z/2 + 1/2z); -im*(z/2 + 1/2z) 0]

function tangentcurvature(X, Y)
    Ẋ,Ẏ = diff(X),diff(Y)
    Ẋ .^ 2 + Ẏ .^ 2
end

T = function(z)
    Xd = Ẋ[z]
    Yd = Ẏ[z]
    N = sqrt(Xd^2 + Yd^2)
    (Xd/N, Yd/N)
end

##
# the joint eigenvectors of Q tell us how to relate T to the 2 tangent vectors
##
z = exp(0.6im)
x,y,Q = jointeigen(X[z],Y[z])

@test (Q'T(z)[1]*Q) ≈ real(Diagonal(Q'T(z)[1]*Q))
@test (Q'T(z)[2]*Q) ≈ real(Diagonal(Q'T(z)[2]*Q))

##
# The curvature is defined as `norm(Ṫ)`. 
# or equivalentally
#
# (ẋ*ÿ - ẏ*ẍ)/(ẋ^2 + ẏ^2)^(3/2)

Ẍ,Ÿ = diff(Ẋ),diff(Ẏ)
κ = function(z)
    Xd = Ẋ[z]; X2 = Ẍ[z]
    Yd = Ẏ[z]; Y2 = Ÿ[z]
    @assert Xd^2 + Yd^2 ≈ real(Xd^2 + Yd^2)
    N = Symmetric(real(Xd^2 + Yd^2))^(3/2)
    (Xd * Y2 - Yd * X2)/N
end


# note the direction of movement of the two points is different
# so the curvature is opposite sign. This is balenced by the direction
# of the normal
@test Q'*κ(exp(0.1im))*Q ≈ Q'*κ(exp(0.5im))*Q ≈ Diagonal([-1,1])

###
# The unit normal is 
# [-ẏ, ẋ]
#
# we weight it by the curvature κ which corrects for choice of orientation
###
θ = range(0,2π;length=1000)
p = plot(cos.(θ), sin.(θ); label=nothing)
scatter!(x,y; label=nothing)
t_x,t_y = real.(diag.(Ref(Q') .* T(z) .* Ref(Q)))
κs = real(diag(Q'*κ(z)*Q))
for k=1:2
    plot!([x[k],x[k]+t_x[k]],[y[k],y[k]+t_y[k]]; arrow=true, label="tangent $k")
end
for k=1:2
    plot!([x[k],x[k]-κs[k]*t_y[k]],[y[k],y[k]+κs[k]*t_x[k]]; arrow=true, label="normal $k")
end; p




H \ (-κ .* Ẏ)


H = HermLaurent{2}()
z = axes(H,1)
h = 0.01

X₀ = hermlaurent(Zeros(2,2), Eye(2)/2) # z -> Eye(2) * (z/2 + 1/2z)
Y₀ = hermlaurent(Zeros(2,2), [0 -1/2; 1/2 0]) #z -> [0 z/2-1/2z; 1/2z-z/2 0]

Xs = [X₀]
Ys = [Y₀]

for _ = 1:100
    X,Y = Xs[end],Ys[end]
    κ = (Ẋ .* Ÿ .- Ẏ .* Ẍ) .* Symmetric.(real.(Ẋ .^2 .+ Ẏ .^2)) .^ (-3/2)
    ΔX = H / H \ (-κ .* Ẏ)
    ΔY = H / H \ (κ .* Ẋ)
    push!(Xs, X + h*ΔX)
    push!(Ys, Y + h*ΔY)
end

function jointplot!(X, Y)
    θ = range(0,2π;length=100)
    z = exp.(im*θ)
    xy = jointeigvals.(X[z], Y[z])
    x,y = first.(xy),last.(xy)
    n = length(x[1])
    for k=1:n
        plot!(getindex.(x,k), getindex.(y,k))
    end
end

p = plot(;legend=false)
jointplot!(Xs[1], Ys[1])
jointplot!(Xs[11], Ys[11])
jointplot!(Xs[21], Ys[21])
jointplot!(Xs[31], Ys[31])
jointplot!(Xs[41], Ys[41])