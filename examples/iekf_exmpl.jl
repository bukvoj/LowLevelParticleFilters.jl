N = 10000 # Number of simulations
Tmax = 25 # Number of time steps

include("../src/LowLevelParticleFilters.jl") # Include the package
using .LowLevelParticleFilters # Use the package
using Random, Distributions
using LinearAlgebra
using Plots
Random.seed!(42) # Setting the seed


# Define the dynamics and measurement models
function example_dynamics(x,u,p,t)
    x
end

function example_measurement(x,u,p,t)
    [atan((x[2]-1.5)/(x[1]-0)), atan((x[2]-0)/(x[1]-0))]
end


# Define the process and measurement noise
σ_ω = 1 * [1.0 0.0; 0.0 1.0] # Process noise Q
σ_v = 1e-4 * [1.0 0.0; 0.0 1.0]# Measurement noise R
procnoise = MvNormal([0.0, 0.0], σ_ω)
measnoise = MvNormal([0.0, 0.0], σ_v)

# Choose the step length for the IEKF
steplength = 0.5


# Allocate memory for the errors
ukferr = zeros(Tmax,N)
ekferr = zeros(Tmax,N)
iekferr = zeros(Tmax,N)

# SIMULATIONS: Run the UKF, EKF and IEKF
println("Running ", N, " simulations...")
for i in 1:N # We are running N simulations
    # Initialize the filters
    x = [1.5,1.5]
    P = 0.1 * [1.0 0.0; 0.0 1.0]
    ukf = UnscentedKalmanFilter(example_dynamics, example_measurement, σ_ω, σ_v, MvNormal(x,P); nu=1, ny=2, p=nothing)
    iekf = IteratedExtendedKalmanFilter(example_dynamics, example_measurement, σ_ω, σ_v, MvNormal(x,P); nu=1, ny=2, p=nothing, step=steplength)
    ekf = ExtendedKalmanFilter(example_dynamics, example_measurement, σ_ω, σ_v, MvNormal(x,P); nu=1, ny=2, p=nothing)
    
    for t in 1:Tmax # We are running Tmax time steps in each simulation
        x = example_dynamics(x,nothing,nothing,t) + rand(procnoise)
        y = example_measurement(x,nothing,nothing,t) + rand(measnoise)

        predict!(ukf,nothing,nothing,t)
        correct!(ukf, nothing,y,nothing,t)

        predict!(ekf,nothing,nothing,t)
        correct!(ekf, nothing,y,nothing,t)

        predict!(iekf,nothing,nothing,t)
        correct!(iekf, nothing,y,nothing,t)

        # save the error
        global ukferr[t,i] = norm(ukf.x - x)
        global ekferr[t,i] = norm(ekf.x - x)
        global iekferr[t,i] = norm(iekf.x - x)
    end
end


# Compute the RMSE
function rms(data::Vector{Float64})
    return sqrt(mean(data .^ 2))
end
for t in 1:Tmax
    println("t = $t, RMSE = ", rms(ukferr[t,:]), " (UKF), ", rms(ekferr[t,:]), " (EKF)", ", ", rms(iekferr[t,:]), " (IEKF)")
end

ukfrms = [rms(ukferr[t,:]) for t in 1:Tmax]
ekfrms = [rms(ekferr[t,:]) for t in 1:Tmax]
iekfrms = [rms(iekferr[t,:]) for t in 1:Tmax]

# Plot the RMSE
plot(1:Tmax, ukfrms, label="UKF", xlabel="Time", ylabel="RMSE", title="RMSE vs Time", lw=2)
plot!(1:Tmax, ekfrms, label="EKF", lw=2)
plot!(1:Tmax, iekfrms, label="IEKF", lw=2)