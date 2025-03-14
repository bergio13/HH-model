#import Pkg
#Pkg.add("Plots")

using Plots

# Gating variable functions
"""
    alphaM(v::Float64, kTemp::Float64) -> Float64
    betaM(v::Float64, kTemp::Float64) -> Float64
    alphaH(v::Float64, kTemp::Float64) -> Float64
    betaH(v::Float64, kTemp::Float64) -> Float64
    alphaN(v::Float64, kTemp::Float64) -> Float64
    betaN(v::Float64, kTemp::Float64) -> Float64

Compute the alpha and beta values for the gating variables of the Hodgkin-Huxley model.

# Arguments
- `v::Float64`: Membrane voltage (mV).
- `kTemp::Float64`: params. correction factor.

# Returns
- `Float64`: Alpha or beta value for the gating variable.
"""

@inline function alphaM(v::Float64, kTemp::Float64)::Float64
    kTemp * (0.1 * (v + 40)) / (1 - exp(-(v + 40) / 10) + eps())
end

@inline function betaM(v::Float64, kTemp::Float64)::Float64
    kTemp * 4 * exp(-(v + 65) / 18)
end

@inline function alphaH(v::Float64, kTemp::Float64)::Float64
    kTemp * 0.07 * exp(-(v + 65) / 20)
end

@inline function betaH(v::Float64, kTemp::Float64)::Float64
    kTemp / (exp(-(v + 35) / 10) + 1)
end

@inline function alphaN(v::Float64, kTemp::Float64)::Float64
    kTemp * (0.01 * (v + 55)) / (1 - exp(-(v + 55) / 10) + eps())
end

@inline function betaN(v::Float64, kTemp::Float64)::Float64
    kTemp * 0.125 * exp(-(v + 65) / 80)
end

# Plotting function
"""
    plot_voltage_trace(time_vector::Vector{Float64}, voltage_vector::Vector{Float64}, m_vector::Vector{Float64},
        h_vector::Vector{Float64}, n_vector::Vector{Float64}, params::HHParameters)

Plot the voltage trace and gating variables of the Hodgkin-Huxley model.
"""
function plot_voltage_trace(time_vector, voltage_vector, m_vector, h_vector, n_vector, params)
    plt = plot(time_vector, voltage_vector, label="Membrane Voltage (mV)", xlabel="Time (ms)", ylabel="Voltage (mV)",
        legend=:topright, xlims=(0, params.simulation_duration), ylims=(-100, 60))
    # Add a filled rectangle for stimulus
    plot!(plt, [params.stimulus_start, params.stimulus_start + params.stimulus_duration,
            params.stimulus_start + params.stimulus_duration, params.stimulus_start, params.stimulus_start],
        [-100, -100, -90, -90, -100],
        fill=true, fillalpha=0.5, lw=0, color=:red, label="")
    plt2 = plot(time_vector, m_vector, label="m", xlabel="Time (ms)", ylabel="Gating Variable",
        legend=:topright, xlims=(0, params.simulation_duration), ylims=(0, 1))
    plot!(time_vector, h_vector, label="h", xlabel="Time (ms)", ylabel="Gating Variable",
        legend=:topright, xlims=(0, params.simulation_duration), ylims=(0, 1))
    plot!(time_vector, n_vector, label="n", xlabel="Time (ms)", ylabel="Gating Variable",
        legend=:topright, xlims=(0, params.simulation_duration), ylims=(0, 1))

    combined_plot = plot(plt, plt2, layout=(1, 2), size=(1000, 400), margin=10Plots.mm)
    display(combined_plot)
end


# Parameters struct
"""
    HHParameters

A struct to hold the parameters of the Hodgkin-Huxley model.
    
# Fields
- `stimulus_current::Float64`: Stimulus current (μA/cm²).
- `stimulus_duration::Float64`: Duration of the stimulus (ms).
- `stimulus_start::Float64`: Time at which the stimulus starts (ms).
- `time_step::Float64`: Time step for numerical inetgration (ms).
- `simulation_duration::Float64`: Total duration of the simulation (ms).
- `temperature::Float64`: params. of the system (°C).
- `membrane_capacitance::Float64`: Membrane capacitance (μF/cm²).
- `solver::String`: ODE solver type, `"FE"` for Forward Euler, `"BE"` for Backward Euler.
- `plot_voltage::Bool`: If `true`, plots the membrane voltage trace.
"""
mutable struct HHParameters
    stimulus_current::Float64
    stimulus_duration::Float64
    stimulus_start::Float64
    time_step::Float64
    simulation_duration::Float64
    temperature::Float64
    membrane_capacitance::Float64
    solver::String
    plot_voltage::Bool
end

# Main function to simulate the HH model
"""
    HH_single(params::HHParameters) -> Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}

Simulates the Hodgkin-Huxley single-compartment model of a neuron.

# Arguments
- `params::HHParameters`: Parameters of the model.

# Returns
A tuple containing:
- `time_vector::Vector{Float64}`: Time points (ms).
- `voltage_vector::Vector{Float64}`: Membrane voltage over time (mV).
- `m_vector::Vector{Float64}`: Activation gating variable for sodium channels.
- `h_vector::Vector{Float64}`: Inactivation gating variable for sodium channels.
- `n_vector::Vector{Float64}`: Activation gating variable for potassium channels.

# Example
```julia

params = HHParameters(20.0, 0.5, 1.0, 0.025, 10.0, 6.3, 1.0, "FE", true)

timeStep, vVec, mVec, hVec, nVec = HH_single(params)
```
"""
function HH_single(params::HHParameters)::Tuple{Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64}}
    # Additional HH parameters
    v_init = -65.0
    gNa = 120.0
    gK = 36.0
    gL = 0.3
    eNa = 50.0
    eK = -77.0
    eL = -54.3

    # Temperature correction factor
    k = 3^((params.temperature - 6.3) / 10)

    # Time vector
    time_steps = Int(params.simulation_duration / params.time_step) + 1
    time_vector = range(0, params.simulation_duration, length=time_steps)

    # Compute initial gating variables
    v0 = v_init
    m0 = alphaM(v0, k) / (alphaM(v0, k) + betaM(v0, k))
    h0 = alphaH(v0, k) / (alphaH(v0, k) + betaH(v0, k))
    n0 = alphaN(v0, k) / (alphaN(v0, k) + betaN(v0, k))

    # Allocate memory for state variables
    voltage_vector = zeros(time_steps)
    m_vector, h_vector, n_vector = zeros(time_steps), zeros(time_steps), zeros(time_steps)

    # Set initial values
    voltage_vector[1] = v0
    m_vector[1], h_vector[1], n_vector[1] = m0, h0, n0

    # Solve ODE and postprocess
    elapsed_time = @elapsed begin
        for t in 1:(time_steps-1)
            vt, mt, ht, nt = voltage_vector[t], m_vector[t], h_vector[t], n_vector[t]

            current_time = time_vector[t]
            iStim = (current_time >= params.stimulus_start && current_time < (params.stimulus_start + params.stimulus_duration)) ? params.stimulus_current : 0.0

            # Ionic currents
            iNa = gNa * mt^3 * ht * (vt - eNa)
            iK = gK * nt^4 * (vt - eK)
            iL = gL * (vt - eL)
            iIon = iNa + iK + iL

            # Update gating variables
            if params.solver == "FE"
                voltage_vector[t+1] = vt + (-iIon + iStim) * (params.time_step / params.membrane_capacitance)
                m_vector[t+1] = mt + (alphaM(vt, k) * (1 - mt) - betaM(vt, k) * mt) * params.time_step
                h_vector[t+1] = ht + (alphaH(vt, k) * (1 - ht) - betaH(vt, k) * ht) * params.time_step
                n_vector[t+1] = nt + (alphaN(vt, k) * (1 - nt) - betaN(vt, k) * nt) * params.time_step
            elseif params.solver == "BE"
                vAdd = 0.001
                rhsdidv = ((gNa * mt^3 * ht * (vt + vAdd - eNa) - iNa) +
                           (gK * nt^4 * (vt + vAdd - eK) - iK) +
                           (gL * (vt + vAdd - eL) - iL)) / vAdd
                voltage_vector[t+1] = (vt + (-iIon + rhsdidv * vt + iStim) * (params.time_step / params.membrane_capacitance)) / (1 + rhsdidv * (params.time_step / params.membrane_capacitance))

                m_vector[t+1] = (mt + params.time_step * alphaM(voltage_vector[t+1], k)) / (1 + params.time_step * (alphaM(voltage_vector[t+1], k) + betaM(voltage_vector[t+1], k)))
                h_vector[t+1] = (ht + params.time_step * alphaH(voltage_vector[t+1], k)) / (1 + params.time_step * (alphaH(voltage_vector[t+1], k) + betaH(voltage_vector[t+1], k)))
                n_vector[t+1] = (nt + params.time_step * alphaN(voltage_vector[t+1], k)) / (1 + params.time_step * (alphaN(voltage_vector[t+1], k) + betaN(voltage_vector[t+1], k)))
            else
                error("Unknown solver type: $(params.solver)")
            end
        end
    end

    println("Simulation completed in $(elapsed_time) seconds.")

    # Plot the voltage trace
    if params.plot_voltage
        plot_voltage_trace(time_vector, voltage_vector, m_vector, h_vector, n_vector, params)
    end

    return time_vector, voltage_vector, m_vector, h_vector, n_vector
end


# Example usage
params = HHParameters(20.0, 0.5, 1.0, 0.025, 10.0, 6.3, 1.0, "BE", true)
timeStep, vVec, mVec, hVec, nVec = HH_single(params)
