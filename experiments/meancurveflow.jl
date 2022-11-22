using AlgebraicCurveOrthogonalPolynomials, ClassicalOrthogonalPolynomials, FillArrays, Test, Plots

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

function jointscatter!(X, Y)
    θ = range(0,2π;length=100)
    z = exp.(im*θ)
    xy = jointeigvals.(X[z], Y[z])
    x,y = first.(xy),last.(xy)
    n = length(x[1])
    for k=1:n
        scatter!(getindex.(x,k), getindex.(y,k))
    end
end


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
    Ẋ,Ẏ = diff(X), diff(Y)
    Ẍ,Ÿ = diff(Ẋ),diff(Ẏ)
    κ = (Ẋ .* Ÿ .- Ẏ .* Ẍ) .* Symmetric.(real.(Ẋ .^2 .+ Ẏ .^2)) .^ (-3/2)
    ΔX = H / H \ (-κ .* Ẏ)
    ΔY = H / H \ (κ .* Ẋ)
    push!(Xs, X + h*ΔX)
    push!(Ys, Y + h*ΔY)
end



p = plot(;legend=false)
for k = 1:10:length(Xs)
    jointplot!(Xs[k], Ys[k])
end; p


###
# cubic perturbed circle
###

Ax = Symmetric(diagm(2 => 0.5*[1,1,1,1]))
Bx = diagm(-4 => 0.5*[1,1])
ε = 0.5
ϕ = asin(ε)/2
c0 = cos(ϕ)/2
c1 = sin(ϕ)/2
c2 = -c0
c3 = -c1
Ay = Symmetric(diagm(1 => [c1,c2,c1,c2,c1],3 => [c0,c3,c0]))
By = diagm(-3 => [c3,c0,c3])
By[6,1] = c2

X = hermlaurent(Ax, Bx)
Y = hermlaurent(Ay, By)


p = plot(); jointplot!(X, Y); p

θ = range(0,2π; length=1000); plot!(cos.(θ), sin.(θ))



H = HermLaurent{6}()
z = axes(H,1)
h = 0.01

X₀ = hermlaurent(Ax, Bx)
Y₀ = hermlaurent(Ay, By)

Xs = [X₀]
Ys = [Y₀]

for k = 1:10
    @show k
    X,Y = Xs[end],Ys[end]
    Ẋ,Ẏ = diff(X), diff(Y)
    Ẍ,Ÿ = diff(Ẋ),diff(Ẏ)
    κ = (Ẋ .* Ÿ .- Ẏ .* Ẍ) .* Symmetric.(real.(Ẋ .^2 .+ Ẏ .^2)) .^ (-3/2)
    ΔX = H / H \ (-κ .* Ẏ)
    ΔY = H / H \ (κ .* Ẋ)
    push!(Xs, X + h*ΔX)
    push!(Ys, Y + h*ΔY)
end


p = plot(;legend=false)
for k = 1:length(Xs)
    jointplot!(Xs[k], Ys[k])
end; p


θ = range(0,2π;length=100); z = exp.(im*θ)
Ẋ,Ẏ = diff(X), diff(Y)

p = plot()



j = 1
p = plot()
for k=1:6
    x,y = jointeigvals(X[z[j]], Y[z[j]])
    t_x,t_y = jointeigvals(Ẋ[z[j]], Ẏ[z[j]])
    plot!([x[k],x[k]+t_x[k]],[y[k],y[k]+t_y[k]]; arrow=true, label="tangent $k")
end; p

κs = real(diag(Q'*κ(z)*Q))
for k=1:2
    plot!([x[k],x[k]-κs[k]*t_y[k]],[y[k],y[k]+κs[k]*t_x[k]]; arrow=true, label="normal $k")
end; p
