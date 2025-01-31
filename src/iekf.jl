function IteratedExtendedKalmanFilter(dynamics, measurement_model::AbstractMeasurementModel, R1,d0=SimpleMvNormal(Matrix(R1)); nu=0, ny=measurement_model.ny, Ts = 1.0, p = NullParameters(), α = 1.0, check = true, Ajac = nothing)
    return ExtendedKalmanFilter(dynamics, measurement_model::AbstractMeasurementModel, R1,d0=SimpleMvNormal(Matrix(R1)); nu=0, ny=measurement_model.ny, Ts = 1.0, p = NullParameters(), α = 1.0, check = true, Ajac = nothing)
end

function IteratedExtendedKalmanFilter(dynamics, measurement, R1,R2,d0=SimpleMvNormal(Matrix(R1)); nu::Int, ny=size(R2,1), Cjac = nothing, step = 1.0, kwargs...)
    IPM = !has_oop(measurement)
    T = promote_type(eltype(R1), eltype(R2), eltype(d0))
    nx = size(R1,1)
    measurement_model = IEKFMeasurementModel{T, IPM}(measurement, R2; nx, ny, Cjac, step)
    return ExtendedKalmanFilter(dynamics, measurement_model, R1, d0; nu, kwargs...)
end

function IteratedExtendedKalmanFilter(kf, dynamics, measurement; Ajac = nothing, Cjac = nothing, step = 1.0)
    IPD = !has_oop(dynamics)
    if measurement isa AbstractMeasurementModel
        measurement_model = measurement
        IPM = isinplace(measurement_model)
    else
        IPM = has_ip(measurement)
        T = promote_type(eltype(kf.R1), eltype(kf.R2), eltype(kf.d0))
        measurement_model = IEKFMeasurementModel{T, IPM}(measurement, kf.R2; kf.nx, kf.ny, Cjac, step)
    end
    if Ajac === nothing
        if IPD
            outx = zeros(eltype(kf.d0), kf.nx)
            jacx = zeros(eltype(kf.d0), kf.nx, kf.nx)
            Ajac = (x,u,p,t) -> ForwardDiff.jacobian!(jacx, (xd,x)->dynamics(xd,x,u,p,t), outx, x)
        else
            Ajac = (x,u,p,t) -> ForwardDiff.jacobian(x->dynamics(x,u,p,t), x)
        end
    end

    return ExtendedKalmanFilter{IPD,IPM,typeof(kf),typeof(dynamics),typeof(measurement_model),typeof(Ajac)}(kf, dynamics, measurement_model, Ajac)
end


function correct!(kf::AbstractKalmanFilter,  measurement_model::IEKFMeasurementModel{IPM}, u, y, p = parameters(kf), t::Real = index(kf); R2 = get_mat(measurement_model.R2, kf.x, u, p, t)) where IPM
    (; x,R) = kf
    (; measurement, Cjac, step) = measurement_model
    
    
    # TODO implement the iterations 
    xi = copy(x)

    maxiters = 10 # TODO make this a parameter
    ϵ = 1e-6 # TODO make this a parameter

    C = zeros(measurement_model.ny, kf.nx)

    i = 1
    while true
        prev = copy(xi)
        C = Cjac(xi, u, p, t)
        if IPM
            e = zeros(length(y))
            measurement(e, xi, u, p, t)
            e .= y .- e
        else
            e = y .- measurement(xi, u, p, t)
        end
        S = symmetrize(C*R*C') + R2
        Sᵪ  = cholesky(Symmetric(S); check=false)
        issuccess(Sᵪ) || error("Cholesky factorization of innovation covariance failed, got S = ", S)
        K = (R*C')/Sᵪ
        xi += step*(x-xi+vec(K)*(e-C*(x-xi)))
        if sum(abs, xi-prev) < ϵ || i >= maxiters
            kf.x = xi
            kf.R = symmetrize((I - K*C)*R) # WARNING against I .- A
            ll = extended_logpdf(SimpleMvNormal(PDMat(S, Sᵪ)), e)[]# - 1/2*logdet(S) # logdet is included in logpdf
            return (; ll, e, S, Sᵪ, K)
        end
        i += 1
    end
end






