using Optim


#(B_x'/z + A_x + B_x *z)^2 + (B_y'/z + A_y + B_y *z)^2 =
# (B_x^2 + B_y^2)'/z^2 + (A_x * B_x' + B_x' * A_x + A_y * B_y' + B_y' * A_y) /z +
# A_x^2 + B_x'*B_x  + B_x*B_x' + A_y^2 + B_y'*B_y  + B_y*B_y' +
# ...

# (B_x'/z + A_x + B_x *z) (B_y'/z + A_y + B_y *z) ==
# B_x'*B_y' / z^2 + (B_x' * A_y + A_x * B_y') /z + B_x' * B_y + A_x * A_y + B_x * B_y'


# 

function unroll(XY_in::AbstractVector)
    XY = reshape(XY_in,8,2)
    A_x = XY[1:2,:]; A_y = XY[3:4,:]; B_x = XY[5:6,:]; B_y = XY[7:8,:];
    Symmetric(A_x), Symmetric(A_y), B_x, B_y
end

function eqns(XY_in)
    A_x, A_y, B_x, B_y = unroll(XY_in)
    norm(B_x^2 + B_y^2)^2 + norm(A_x * B_x' + B_x' * A_x + A_y * B_y' + B_y' * A_y)^2 +
        norm(A_x^2 + B_x'*B_x  + B_x*B_x' + A_y^2 + B_y'*B_y  + B_y*B_y' - I)^2 +
        norm(B_x'*B_y' - B_y'*B_x')^2 + norm(B_x'*A_y + A_x * B_y' - (B_y'*A_x + A_y * B_x'))^2 +
        norm(( B_x' * B_y + A_x * A_y + B_x * B_y') - ( B_y' * B_x + A_y * A_x + B_y * B_x'))^2 +
        norm(B_x' + A_x + B_x - I)^2 + norm(B_y' + A_y + B_y)^2 + 
        norm(-B_x' + A_x - B_x + I)^2 + norm(-B_y' + A_y - B_y)^2
end

A_x = [0 0; 0 0]
B_x = [0.5 0; 0 0.5]
A_y = [0 0; 0 0]
B_y = [0 -0.5; 0.5 0]

X = z -> (B_x')/z + A_x + B_x*z
Y = z -> (B_y')/z + A_y + B_y*z
X(-1)
Y(1)

XY = [A_x; A_y; B_x; B_y]

eqns(XY)


function eqns(A_x, A_y, B_x, B_y)
    norm(B_x^2 + B_y^2)^2 + norm(A_x * B_x' + B_x' * A_x + A_y * B_y' + B_y' * A_y)^2 +
        norm(A_x^2 + B_x'*B_x  + B_x*B_x' + A_y^2 + B_y'*B_y  + B_y*B_y' - I)^2 +
        norm(B_x'*B_y' - B_y'*B_x')^2 + norm(B_x'*A_y + A_x * B_y' - (B_y'*A_x + A_y * B_x'))^2 +
        norm(( B_x' * B_y + A_x * A_y + B_x * B_y') - ( B_y' * B_x + A_y * A_x + B_y * B_x'))^2 +
        norm(B_x' + A_x + B_x - I)^2 + norm(B_y' + A_y + B_y)^2 + 
        norm(-B_x' + A_x - B_x + I)^2 + norm(-B_y' + A_y - B_y)^2
end


function eqns(A_x, A_y, B_x, B_y)
    norm(B_x^2 + B_y^2)^2 + norm(A_x * B_x' + B_x' * A_x + A_y * B_y' + B_y' * A_y)^2 +
        norm(A_x^2 + B_x'*B_x  + B_x*B_x' + A_y^2 + B_y'*B_y  + B_y*B_y' - I)^2 +
        norm(B_x'*B_y' - B_y'*B_x')^2 + norm(B_x'*A_y + A_x * B_y' - (B_y'*A_x + A_y * B_x'))^2 +
        norm(( B_x' * B_y + A_x * A_y + B_x * B_y') - ( B_y' * B_x + A_y * A_x + B_y * B_x'))^2 +
        norm(B_x' + A_x + B_x - I)^2 + norm(B_y' + A_y + B_y)^2 + 
        norm(-B_x' + A_x - B_x)^2 + norm(-B_y' + A_y - B_y + [0 1; 1 0])^2
end


eqns(XY_in) = eqns(unroll(XY_in)...)

result = optimize(eqns, randn(16), Newton(); autodiff=:forward)
A_x, A_y, B_x, B_y = unroll(Optim.minimizer(result))

A_x, A_y, B_x, B_y = [1/2 0; 0 1/2], [0 -1/2; -1/2 0], [1/4 0; 0 1/4], [0 (1+sqrt(2))/4; (1-sqrt(2))/4 0]
eqns(A_x, A_y, B_x, B_y)


Q = qr(randn(2,2)).Q; eqns(Q'*A_x*Q, Q'*A_y*Q, Q'*B_x*Q, Q'*B_y*Q)