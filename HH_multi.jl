using LinearAlgebra
using Printf
using Plots

### ------------- DEFINE FUNCTIONS FOR ALPHAS AND BETAS -------------

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


"""
    HH_multi(; I=-5, nComp=201, rhoA=100, cellX=0, plotV=true)

Simulates the Hodgkin-Huxley multi-compartment model of a neuron.

# Arguments
- `I::Float64=-5`: Stimulus current (μA/cm²).
- `nComp::Int=201`: Number of compartments in the model.
- `rhoA::Float64=100`: Axial resistivity (Ωcm).
- `cellX::Float64=0`: X-shift of the cell relative to the electrode (at 0/0) (μm).
- `plotV::Bool=true`: Whether to plot the voltage propagation.

# Returns
- `vMat::Array{Float64,2}`: Matrix of membrane voltages for each compartment over time.
- `mMat::Array{Float64,2}`: Matrix of m-gating variables for each compartment over time.
- `hMat::Array{Float64,2}`: Matrix of h-gating variables for each compartment over time.
- `nMat::Array{Float64,2}`: Matrix of n-gating variables for each compartment over time.
- `iNaMat::Array{Float64,2}`: Matrix of sodium currents for each compartment over time.
- `iKMat::Array{Float64,2}`: Matrix of potassium currents for each compartment over time.
- `iLMat::Array{Float64,2}`: Matrix of leak currents for each compartment over time.

"""
function HH_multi(; I=-5, nComp=201, rhoA=100, cellX=0, plotV=true)

    # Electrode and spatial parameters
    rhoE = 300  # Extracellular resistivity, in Ohm*cm, for 'point and 'disk' only
    cellY = 25  # Y-shift of cell relative to the electrode (at 0/0)), in um

    # Temporal parameters
    tStop = 10  # Total duration of simulation, in ms
    tDel = 1    # Time when stimulus starts, in ms
    tDur = 0.5  # Time of stimulus ON, in ms
    tDt = 0.0125 # Time step, in ms

    # Stick/fiber parameters
    lStick = 2000  # Total length of stick/fiber, in um
    rComp = 1  # Compartment radius, in um
    # Biophysics
    c = 1  # Membrane specific capacitance, in uF/cm2
    # Temperature
    temp = 6.3  # Model temperature, in Celsius, original=6.3

    # Hodgkin & Huxley parameters
    vInit = -65  # Membrane voltage to initialize the simulation, in mV
    gNa = 120    # Sodium channel maximum conductivity, in mS/cm2, original=120
    gK = 36      # Potassium channel maximum conductivity, in mS/cm2, original=36
    gL = 0.3     # Leak channel maximum conductivity, in mS/cm2, original=0.3
    eNa = 50     # Sodium reversal/equilibrium potential, in mV, original=50
    eK = -77     # Potassium reversal/equilibrium potential, in mV, original=-77
    eL = -54.3   # Leak reversal/equilibrium potential, in mV, original=-54.3

    lComp = lStick / (nComp - 1)
    # Central compartment of stick - Fixed Integer division
    centerComp = div(nComp - 1, 2) + 1

    # Axial resistance of each compartment, in kOhm
    R = ((2 * rhoA * lComp * 1e-4) / (2 * (rComp * 1e-4)^2 * π)) * 1e-3
    # Compartment surface, in cm2
    A = (2 * rComp * π * lComp) * 1e-8
    # Compartment membrane capacitance, in uF
    C = c * A

    # Time step
    timeSteps = Int(tStop / tDt) + 1
    timeStep = range(0, tStop, length=timeSteps)

    # Temperature adjustment
    kTemp = 3^((temp - 6.3) / 10)

    # Other constants
    vAdd = 0.001

    # Set up tridiagonal matrix for fast computation of iAxial
    matInvDiag = vcat(
        [1 + (tDt / C) * (1 / R)],   # First element
        ones(nComp - 2) .+ (tDt / C) .* (2 / R),  # Middle elements
        [1 + (tDt / C) * (1 / R)]    # Last element
    )
    matInvOffDiag = ones(nComp - 1) .* (-(tDt / C) * (1 / R))
    matInv = Tridiagonal(matInvOffDiag, matInvDiag, matInvOffDiag)

    # Set up matrix
    matAxDiag = vcat([-1 / R], ones(nComp - 2) .* (-2 / R), [-1 / R])  # in 1/kOhm
    matAxOffDiag = ones(nComp - 1) .* (1 / R)
    matAxial = Tridiagonal(matAxOffDiag, matAxDiag, matAxOffDiag)

    x = range(-(centerComp - 1) * lComp, (centerComp - 1) * lComp, length=nComp) .+ cellX
    y = ones(nComp) .* cellY

    # Euklidean distance for each compartment center
    compDist = 1e-4 .* sqrt.(x .^ 2 .+ y .^ 2)  # in cm
    # Analytical potentials (point source) for given distance
    potentials = 1e-3 .* (rhoE * I) ./ (4 * π .* compDist)  # in mV
    potentials = reshape(potentials, :, 1)


    iStim = (matAxial * potentials) ./ A  # in μA/cm²


    v0 = vInit
    m0 = alphaM(v0, kTemp) / (alphaM(v0, kTemp) + betaM(v0, kTemp))
    h0 = alphaH(v0, kTemp) / (alphaH(v0, kTemp) + betaH(v0, kTemp))
    n0 = alphaN(v0, kTemp) / (alphaN(v0, kTemp) + betaN(v0, kTemp))

    # Allocate memory for v, m, h and n
    vMat = zeros(nComp, timeSteps)
    mMat = zeros(nComp, timeSteps)
    hMat = zeros(nComp, timeSteps)
    nMat = zeros(nComp, timeSteps)

    # Set initial values
    vMat[:, 1] .= v0
    mMat[:, 1] .= m0
    hMat[:, 1] .= h0
    nMat[:, 1] .= n0


    ### --------- Step 3: SOLVE ODE & POSTPROCESSING ---------
    Tic = time()
    for t in 1:(timeSteps-1)

        # States at current time step
        vVecT = vMat[:, t]
        mVecT = mMat[:, t]
        hVecT = hMat[:, t]
        nVecT = nMat[:, t]

        # Stimulus current
        iStimVec = zeros(nComp)
        if t >= Int(tDel / tDt) && t < Int((tDel + tDur) / tDt)
            iStimVec = iStim  # in uA/cm2
        end

        # Ionic currents
        # Sodium
        iNaVec = gNa .* mVecT .^ 3 .* hVecT .* (vVecT .- eNa)  # in (mS/cm2)*mV==uA/cm2
        # Potassium
        iKVec = gK .* nVecT .^ 4 .* (vVecT .- eK)  # in (mS/cm2)*mV==uA/cm2
        # Leak
        iLVec = gL .* (vVecT .- eL)  # in (mS/cm2)*mV==uA/cm2
        # Sum
        iIonVec = iNaVec .+ iKVec .+ iLVec  # in uA/cm2    

        # Additional ionic contribution needed for BE
        # Sodium
        iNaAuxVec = gNa .* (mVecT .^ 3) .* hVecT .* (vVecT .+ vAdd .- eNa)  # in (mS/cm2)*mV == uA/cm2
        # Potassium
        iKAuxVec = gK .* (nVecT .^ 4) .* (vVecT .+ vAdd .- eK)  # in (mS/cm2)*mV == uA/cm2
        # Leak
        iLAuxVec = gL .* (vVecT .+ vAdd .- eL)  # in (mS/cm2)*mV == uA/cm2
        # Sum
        rhsdidvVec = ((iNaAuxVec .- iNaVec) .+ (iKAuxVec .- iKVec) .+ (iLAuxVec .- iLVec)) ./ vAdd

        # Compute change of v
        # Right-hand side of matrix equation to be solved
        RHS = vVecT .+ ((-iIonVec .+ rhsdidvVec .* vVecT .+ iStimVec) .* (tDt / c))

        # Create a copy of matInv for this timestep
        matInvCopy = copy(matInv)

        # Add ionic current contribution to left-hand side
        for i in 1:nComp
            matInvCopy[i, i] = matInvDiag[i] + rhsdidvVec[i] * (tDt / c)
        end

        # Solve matrix equation
        vMat[:, t+1] = matInvCopy \ RHS  # `\` is used for solving Ax = b

        # Update gating variables with new v
        mMat[:, t+1] = (mVecT .+ tDt .* alphaM.(vMat[:, t+1], kTemp)) ./ (1 .+ tDt .* (alphaM.(vMat[:, t+1], kTemp) .+ betaM.(vMat[:, t+1], kTemp)))
        hMat[:, t+1] = (hVecT .+ tDt .* alphaH.(vMat[:, t+1], kTemp)) ./ (1 .+ tDt .* (alphaH.(vMat[:, t+1], kTemp) .+ betaH.(vMat[:, t+1], kTemp)))
        nMat[:, t+1] = (nVecT .+ tDt .* alphaN.(vMat[:, t+1], kTemp)) ./ (1 .+ tDt .* (alphaN.(vMat[:, t+1], kTemp) .+ betaN.(vMat[:, t+1], kTemp)))
    end

    Toc = time()
    @printf("--- Solving time was %.3f seconds\n", Toc - Tic)

    # Compute ionic current densities using state variables v, m, h, and n
    iNaMat = gNa .* mMat .^ 3 .* hMat .* (vMat .- eNa)  # in (mS/cm²) * mV == uA/cm²
    iKMat = gK .* nMat .^ 4 .* (vMat .- eK)  # in (mS/cm²) * mV == uA/cm²
    iLMat = gL .* (vMat .- eL)  # in (mS/cm²) * mV == uA/cm²

    if plotV

        # Define main voltage plot
        p = plot(title="Hodgkin-Huxley Model - All Compartments",
            xlabel="Time (ms)", ylabel="Voltage (mV)",
            titlefontsize=12, guidefontsize=10, tickfontsize=8,
            legend=false, grid=true)

        # Plot all compartments in light blue with some transparency
        for i in 1:nComp
            if i != centerComp
                plot!(p, timeStep, vMat[i, :], color=:lightblue, alpha=0.5, linewidth=1)
            end
        end

        # Highlight the central compartment in red
        plot!(p, timeStep, vMat[centerComp, :], color=:red, linewidth=3, label="Central Compartment")

        # Plot ionic currents
        p2 = plot(title="Ionic Currents at Central Compartment",
            xlabel="Time (ms)", ylabel="Current (μA/cm²)",
            titlefontsize=12, guidefontsize=10, tickfontsize=8,
            legend=:topright, grid=true)

        plot!(p2, timeStep, iNaMat[centerComp, :], label="Na+ Current", linewidth=2, color=:blue)
        plot!(p2, timeStep, iKMat[centerComp, :], label="K+ Current", linewidth=2, color=:green)
        plot!(p2, timeStep, iLMat[centerComp, :], label="Leak Current", linewidth=2, color=:black)
        plot!(p2, timeStep, iNaMat[centerComp, :] .+ iKMat[centerComp, :] .+ iLMat[centerComp, :],
            label="Total Ionic Current", linewidth=2, color=:purple, linestyle=:dash)

        # Heatmap of voltage propagation
        p3 = heatmap(timeStep, 1:nComp, vMat,
            title="Voltage Propagation in Neural Fiber",
            xlabel="Time (ms)", ylabel="Compartment",
            titlefontsize=12, guidefontsize=10, tickfontsize=8,
            color=:plasma, colorbar_title="Voltage (mV)", aspect_ratio=:auto)

        # Highlight the central compartment in the heatmap
        hline!([centerComp], color=:red, linewidth=2, label="Central Compartment")

        # First two plots side by side, heatmap below
        layout = @layout [a b; c]

        # Combine and display
        combined_plot = plot(p, p2, p3, layout=layout, size=(1600, 1000), margin=10Plots.mm)
        display(combined_plot)
        # Save the plot
        #savefig(combined_plot, "hodgkin_huxley_plots.png")

    end
    return vMat, mMat, hMat, nMat, iNaMat, iKMat, iLMat
end

# Run the model
vMat, mMat, hMat, nMat, iNaMat, iKMat, iLMat = HH_multi(I=-5)