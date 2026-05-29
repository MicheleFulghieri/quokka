//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testCosmologicalDarkMatter.cpp
/// \brief Defines a cosmology problem for the power spectrum evolution


#include "QuokkaSimulation.hpp"
#include "particles/particle_types.hpp"
#include "cosmology/Cosmology.hpp"

#include "AMReX_BLassert.H"
#include <AMReX_ParticleMesh.H>

#include <cmath>

struct CosmoPowerSpectrum {
};


// NB: in CAMB messo H0=67.5
// z_start = 99



auto problem_main() -> int {
    QuokkaSimulation<CosmoPowerSpectrum> sim;
    sim.setInitialConditions();
	sim.evolve();

    return 0;
}