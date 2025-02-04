abstract type AbstractExtendedKalmanFilter{IPD,IPM} <: AbstractKalmanFilter end
@with_kw struct ExtendedKalmanFilter{IPD, IPM, KF <: KalmanFilter, F, G, A, C} <: AbstractExtendedKalmanFilter{IPD,IPM}
    kf::KF
    dynamics::F
    measurement::G
    Ajac::A
    Cjac::C
end

"""
    ExtendedKalmanFilter(kf, dynamics, measurement; Ajac, Cjac)
    ExtendedKalmanFilter(dynamics, measurement, R1,R2,d0=MvNormal(Matrix(R1)); nu::Int, p = NullParameters(), α = 1.0, check = true)

A nonlinear state estimator propagating uncertainty using linearization.

The constructor to the extended Kalman filter takes dynamics and measurement functions, and either covariance matrices, or a [`KalmanFilter`](@ref). If the former constructor is used, the number of inputs to the system dynamics, `nu`, must be explicitly provided with a keyword argument.

By default, the filter will internally linearize the dynamics using ForwardDiff. User provided Jacobian functions can be provided as keyword arguments `Ajac` and `Cjac`. These functions should have the signature `(x,u,p,t)::AbstractMatrix` where `x` is the state, `u` is the input, `p` is the parameters, and `t` is the time.

The dynamics and measurement function are on the following form
```
x(t+1) = dynamics(x, u, p, t) + w
y      = measurement(x, u, p, t) + e
```
where `w ~ N(0, R1)`, `e ~ N(0, R2)` and `x(0) ~ d0`

See also [`UnscentedKalmanFilter`](@ref) which is typically more accurate than `ExtendedKalmanFilter`. See [`KalmanFilter`](@ref) for detailed instructions on how to set up a Kalman filter `kf`.
"""
ExtendedKalmanFilter

function ExtendedKalmanFilter(dynamics, measurement, R1,R2,d0=SimpleMvNormal(Matrix(R1)); nu::Int, ny=nothing, Ts = 1.0, p = NullParameters(), α = 1.0, check = true, Ajac = nothing, Cjac = nothing)
    nx = size(R1,1)
    ny = size(R2,1)
    T = eltype(R1)
    if R1 isa SMatrix
        x = @SVector zeros(T, nx)
        u = @SVector zeros(T, nu)
    else
        x = zeros(T, nx)
        u = zeros(T, nu)
    end
    t = zero(T)
    A = zeros(nx, nx) # This one is never needed
    B = zeros(nx, nu) # This one is never needed
    C = zeros(ny, nx) # This one is never needed
    D = zeros(ny, nu) # This one is never needed
    kf = KalmanFilter(A,B,C,D,R1,R2,d0; Ts, p, α, check)

    return ExtendedKalmanFilter(kf, dynamics, measurement; Ajac, Cjac)
end

function ExtendedKalmanFilter(kf, dynamics, measurement; Ajac = nothing, Cjac = nothing)
    IPD = has_ip(dynamics)
    IPM = has_ip(measurement)
    if Ajac === nothing
        # if IPD
        #     inner! = (xd,x)->dynamics(xd,x,u,p,t)
        #     out = zeros(eltype(kf.d0), length(kf.x))
        #     cfg = ForwardDiff.JacobianConfig(inner!, out, x)
        #     Ajac = (x,u,p,t) -> ForwardDiff.jacobian!((xd,x)->dynamics(xd,x,u,p,t), out, x, cfg, Val(false))
        # else
        #     inner = x->dynamics(x,u,p,t)
        #     cfg = ForwardDiff.JacobianConfig(inner, kf.x)
        #     Ajac = (x,u,p,t) -> ForwardDiff.jacobian(x->dynamics(x,u,p,t), x, cfg, Val(false))
        # end

        if IPD
            outx = zeros(eltype(kf.d0), kf.nx)
            jacx = zeros(eltype(kf.d0), kf.nx, kf.nx)
            Ajac = (x,u,p,t) -> ForwardDiff.jacobian!(jacx, (xd,x)->dynamics(xd,x,u,p,t), outx, x)
        else
            Ajac = (x,u,p,t) -> ForwardDiff.jacobian(x->dynamics(x,u,p,t), x)
        end
    end
    if Cjac === nothing
        if IPM
            outy = zeros(eltype(kf.d0), kf.ny)
            jacy = zeros(eltype(kf.d0), kf.ny, kf.nx)
            Cjac = (x,u,p,t) -> ForwardDiff.jacobian!(jacy, (y,x)->measurement(y,x,u,p,t), outy, x)
        else
            Cjac = (x,u,p,t) -> ForwardDiff.jacobian(x->measurement(x,u,p,t), x)
        end
    end
    return ExtendedKalmanFilter{IPD,IPM,typeof(kf),typeof(dynamics),typeof(measurement),typeof(Ajac),typeof(Cjac)}(kf, dynamics, measurement, Ajac, Cjac)
end

function Base.getproperty(ekf::EKF, s::Symbol) where EKF <: AbstractExtendedKalmanFilter
    s ∈ fieldnames(EKF) && return getfield(ekf, s)
    return getproperty(getfield(ekf, :kf), s)
end

function Base.setproperty!(ekf::ExtendedKalmanFilter, s::Symbol, val)
    s ∈ fieldnames(typeof(ekf)) && return setproperty!(ekf, s, val)
    setproperty!(getfield(ekf, :kf), s, val) # Forward to inner filter
end

function Base.propertynames(ekf::EKF, private::Bool=false) where EKF <: AbstractExtendedKalmanFilter
    return (fieldnames(EKF)..., propertynames(ekf.kf, private)...)
end


function predict!(kf::AbstractExtendedKalmanFilter{IPD}, u, p = parameters(kf), t::Real = index(kf)*kf.Ts; R1 = get_mat(kf.R1, kf.x, u, p, t), α = kf.α) where IPD
    @unpack x,R = kf
    A = kf.Ajac(x, u, p, t)
    if IPD
        xp = similar(x)
        kf.dynamics(xp, x, u, p, t)
        kf.x = xp
    else
        kf.x = kf.dynamics(x, u, p, t)
    end
    if α == 1
        kf.R = symmetrize(A*R*A') + R1
    else
        kf.R = symmetrize(α*A*R*A') + R1
    end
    kf.t += 1
end

function correct!(kf::AbstractExtendedKalmanFilter{<:Any, IPM}, u, y, p = parameters(kf), t::Real = index(kf); R2 = get_mat(kf.R2, kf.x, u, p, t)) where IPM
    @unpack x,R = kf
    C = kf.Cjac(x, u, p, t)
    if IPM
        e = zeros(length(y))
        kf.measurement(e, x, u, p, t)
        e .= y .- e
    else
        e = y .- kf.measurement(x, u, p, t)
    end
    S   = symmetrize(C*R*C') + R2
    Sᵪ  = cholesky(Symmetric(S); check=false)
    issuccess(Sᵪ) || error("Cholesky factorization of innovation covariance failed, got S = ", S)
    K   = (R*C')/Sᵪ
    kf.x += vec(K*e)
    kf.R  = symmetrize((I - K*C)*R) # WARNING against I .- A
    ll = extended_logpdf(SimpleMvNormal(PDMat(S, Sᵪ)), e)[]# - 1/2*logdet(S) # logdet is included in logpdf
    (; ll, e, S, Sᵪ, K)
end


function smooth(sol, kf::AbstractExtendedKalmanFilter, u::AbstractVector=sol.u, y::AbstractVector=sol.y, p=parameters(kf))
    T            = length(y)
    (; x,xt,R,Rt,ll) = sol
    xT           = similar(xt)
    RT           = similar(Rt)
    xT[end]      = xt[end]      |> copy
    RT[end]      = Rt[end]      |> copy
    for t = T-1:-1:1
        A = kf.Ajac(xT[t+1],u[t+1],p,((t+1)-1)*kf.Ts)
        C     = Rt[t]*A'/R[t+1]
        xT[t] = C*(xT[t+1] .- x[t+1]) .+= xt[t]
        RT[t] = symmetrize(C*(RT[t+1] .- R[t+1])*C') .+= Rt[t]
    end
    xT,RT,ll
end


function smooth(kf::AbstractExtendedKalmanFilter, args...)
    reset!(kf)
    sol = forward_trajectory(kf, args...)
    smooth(sol, kf, args...)
end

sample_state(kf::AbstractExtendedKalmanFilter, p=parameters(kf); noise=true) = noise ? rand(kf.d0) : mean(kf.d0)
sample_state(kf::AbstractExtendedKalmanFilter, x, u, p, t; noise=true) = kf.dynamics(x, u, p, t) .+ noise*rand(SimpleMvNormal(get_mat(kf.R1, x, u, p, t)))
sample_measurement(kf::AbstractExtendedKalmanFilter, x, u, p, t; noise=true) = kf.measurement(x, u, p, t) .+ noise*rand(SimpleMvNormal(get_mat(kf.R2, x, u, p, t)))
measurement(kf::AbstractExtendedKalmanFilter) = kf.measurement
dynamics(kf::AbstractExtendedKalmanFilter) = kf.dynamics