# Hodgkin-Huxley Neuron Model in Julia

## Overview

This Julia script implements the Hodgkin-Huxley (HH) model to simulate the electrical activity of a neuron. The HH model describes how action potentials in neurons are initiated and propagated through voltage-gated ion channels.

The model simulates the membrane potential over time, taking into account sodium (Na⁺), potassium (K⁺), and leak (L) ion channels, as well as an external stimulus current.

The script includes two implementations of the HH model: one for a single compartment and one for multiple compartments.

## Contents

- `HH_single.jl` contains the implementation of the HH model for a single compartment
- `HH_multi.jl` contains the implementation of the HH model for multiple compartments

## Results

### Single Compartment

![Hodgkin-Huxley Model Simulation (Single Compartment)](https://github.com/bergio13/hh_model/blob/main/images/plot.png?raw=true)

### Multi Compartment

![Hodgkin-Huxley Model Simulation (Multi Compartment)](https://github.com/bergio13/hh_model/blob/main/images/multi_plots.png?raw=true)
