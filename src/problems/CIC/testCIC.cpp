//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testCIC.cpp
/// \brief Defines a test problem for the CIC particles in view of adding cosmology.
///

#include "Cosmology.hpp"
#include "QuokkaSimulation.hpp"

struct DarkMatter {
	// empty struct, tag for the templates: all the following functions follow the testCIC configurations
};

template <> struct quokka::EOS_Traits<DarkMatter> {
	/* data */
};
