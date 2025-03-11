# Hodgkin-Huxley Neuron Model in Julia

## Overview

This Julia script implements the Hodgkin-Huxley (HH) model to simulate the electrical activity of a neuron. The HH model describes how action potentials in neurons are initiated and propagated through voltage-gated ion channels.

The model simulates the membrane potential over time, taking into account sodium (Na⁺), potassium (K⁺), and leak (L) ion channels, as well as an external stimulus current.

## Visualization

Setting the `plot` variable to `true` will display a plot of the membrane potential over time. The plot will show the membrane potential and the stimulus current over time.

![Hodgkin-Huxley Model Simulation](images/single.png)
