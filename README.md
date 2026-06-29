# RePU Rate Model

A biologically inspired recurrent **Rectified Polynomial Unit (RePU)** rate model implementing continuous online learning using **strictly synapse-local plasticity**.

## Overview

This repository presents a constructive demonstration that a fully recurrent RePU (SSN-style) network can learn using only information locally available at each synapse. Every plasticity rule depends exclusively on quantities accessible to the pre- and postsynaptic neurons (or their compartments), such as firing rates, local activity statistics, synaptic weight amplitudes, and synapse-specific traces. No global optimization, backpropagation, or other non-local learning mechanisms are used.

The model sits at the intersection of computational neuroscience, machine learning, and adaptive control theory. While biologically inspired, many of the learning rules were developed from a control-theoretic perspective, with an emphasis on maintaining stable adaptive dynamics in highly recurrent networks.

## Current Results

The current implementation demonstrates that a purely synapse-local learning system can simultaneously exhibit:

- Balanced excitatory and inhibitory network dynamics.
- Continuous online learning without separate training and inference phases.
- Competitive, selective feedforward receptive fields emerging through recurrent interactions.
- Sparse, local excitatory and inhibitory connectivity, with inhibitory projections operating over shorter spatial ranges than excitatory projections.
- High recurrent amplification while empirically remaining within a semi-stable operating regime.
- Continuously adapting receptive fields rather than permanently fixed representations.

Several aspects of the observed dynamics—including the semi-stable operating regime—are empirical observations of the current implementation rather than theoretical guarantees.

## Distinguishing Features

Compared with a conventional Stabilized Supralinear Network (SSN), this model explores several additional mechanisms:

- Continuous plasticity throughout network operation.
- Strictly synapse-local learning rules for every adaptive process.
- Continuous multiplicative **L1 normalization** of all synaptic weights within each neuron compartment.
- Homeostatic learning of each compartment's total synaptic weight amplitude (the target L1 norm), allowing overall synaptic resources to adapt over time.
- Sparse local connectivity for all synapse types.
- Learning dynamics that avoid non-local optimization methods such as backpropagation or BPTT.

## Project Status

This repository is an active work in progress.

The implementation is functional, but the documentation, experiments, and code organization are still evolving. A paper describing the learning rules, stability mechanisms, and emergent behaviors is currently in preparation.

For a more detailed discussion of the learning mechanisms and the adaptive control problems they address, see:

```text
LatexDraft/Bio_Constrained_RNN.tex
```

More complete documentation, reproducible experiments, and easier-to-follow demonstration scripts will be added as the manuscript matures.
