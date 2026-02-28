// ABOUTME: Cosmology module for Quokka: background evolution and comoving source terms.
#ifndef COSMOLOGY_HPP_
#define COSMOLOGY_HPP_

//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file Cosmology.hpp
/// \brief Defines the Cosmology module for background evolution and source terms.

/// ## Physics background
/// The cosmological scale factor a(t) evolves according to the Friedmann equation:
///   da/dt = a * H(a)
/// where H(a) is the Hubble parameter:
///   H(a) = H0 * E(a)
///   E(a) = sqrt( Omega_r/a^4 + Omega_m/a^3 + Omega_k/a^2 + Omega_L )
///
/// In comoving coordinates, the hydrodynamics equations pick up source terms:
/// 1. Momentum decay (Hubble drag):
///    d(p_i)/dt = -H * p_i
///    Analytic solution: p_i(t) = p_i(0) * (a_0 / a(t))
///
/// 2. Expansion cooling:
///    d(E_int)/dt = -2 * (gamma - 1) * H * E_int
///    Analytic solution: E_int(t) = E_int(0) * (a_0 / a(t))^[2*(gamma - 1)]

#include <cmath>

#include "AMReX.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "physics_info.hpp"

namespace quokka::cosmology {

/// @brief Parameters for the cosmological model (LCDM by default)
struct CosmologyParams {
	amrex::Real H0{C::Hubble_const}; ///< Hubble constant at z=0 [s^-1]
	amrex::Real Omega_m{0.315};	  ///< Matter density parameter
	amrex::Real Omega_r{9.2618e-5};  ///< Radiation density parameter
	amrex::Real Omega_L{0.685};	  ///< Dark energy (Lambda) density parameter
	// Omega_k = 1 - (Omega_m + Omega_r + Omega_L)
};

/// @brief Compute the dimensionless Hubble factor E(a) = H(a)/H0
/// @param a Scale factor
/// @param cosmo Cosmology parameters
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto HubbleFactor(amrex::Real a, CosmologyParams const &cosmo) -> amrex::Real
{
	const amrex::Real Omega_k = 1.0 - cosmo.Omega_r - cosmo.Omega_m - cosmo.Omega_L;
	const amrex::Real a2 = a * a;
	const amrex::Real a3 = a2 * a;
	const amrex::Real a4 = a2 * a2;

	const amrex::Real E2 = cosmo.Omega_r / a4 + cosmo.Omega_m / a3 + Omega_k / a2 + cosmo.Omega_L;
	return std::sqrt(std::max(E2, static_cast<amrex::Real>(0.0)));
}

/// @brief Evolve the scale factor a(t) from a_old over time dt
/// @param a_old Starting scale factor
/// @param dt Time step
/// @param cosmo Cosmology parameters
/// @param max_frac_step Maximum fractional change in a per sub-step (for accuracy)
/// @return New scale factor a_new
[[nodiscard]] inline auto evolveScaleFactor(amrex::Real a_old, amrex::Real dt, CosmologyParams const &cosmo,
					      amrex::Real max_frac_step = 0.01) -> amrex::Real
{
	const amrex::Real H_est = cosmo.H0 * HubbleFactor(a_old, cosmo);
	// estimate number of sub-steps needed for accuracy
	const int nsteps = std::max(1, static_cast<int>(std::ceil(H_est * dt / max_frac_step)));
	const amrex::Real dt_sub = dt / static_cast<amrex::Real>(nsteps);

	amrex::Real a = a_old;
	for (int step = 0; step < nsteps; ++step) {
		// Midpoint method (RK2)
		const amrex::Real k1 = a * cosmo.H0 * HubbleFactor(a, cosmo);
		const amrex::Real a_mid = a + 0.5 * dt_sub * k1;
		const amrex::Real k2 = a_mid * cosmo.H0 * HubbleFactor(a_mid, cosmo);
		a += dt_sub * k2;
	}
	return a;
}

/// @brief Apply cosmological source terms (Hubble drag and expansion cooling)
/// @tparam problem_t Simulation problem type
/// @param state Cell-centered conserved state MultiFab
/// @param a_old Scale factor at beginning of step
/// @param a_new Scale factor at end of step
template <typename problem_t> void applyCosmologicalSourceTerms(amrex::MultiFab &state, amrex::Real a_old, amrex::Real a_new)
{
	const amrex::Real ratio = a_old / a_new;
	const amrex::Real gamma = quokka::EOS_Traits<problem_t>::gamma;
	const amrex::Real eint_ratio = std::pow(ratio, 3.0 * (gamma - 1.0));

	const int density_idx = HydroSystem<problem_t>::density_index;
	const int px_idx = HydroSystem<problem_t>::x1Momentum_index;
	const int py_idx = HydroSystem<problem_t>::x2Momentum_index;
	const int pz_idx = HydroSystem<problem_t>::x3Momentum_index;
	const int etot_idx = HydroSystem<problem_t>::energy_index;
	const int eint_idx = HydroSystem<problem_t>::internalEnergy_index;

	for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
		const amrex::Box &bx = mfi.validbox();
		auto const &state_arr = state.array(mfi);

		amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const amrex::Real rho = state_arr(i, j, k, density_idx);
			if (rho <= 0.0) {
				return;
			}

			// Update momentum (Hubble drag)
			const amrex::Real px_new = state_arr(i, j, k, px_idx) * ratio;
			const amrex::Real py_new = state_arr(i, j, k, py_idx) * ratio;
			const amrex::Real pz_new = state_arr(i, j, k, pz_idx) * ratio;

			// Update internal energy (Expansion cooling)
			const amrex::Real eint_new = state_arr(i, j, k, eint_idx) * eint_ratio;

			// Update total energy (recompute to maintain consistency)
			const amrex::Real KE_new = 0.5 * (px_new * px_new + py_new * py_new + pz_new * pz_new) / rho;
			const amrex::Real etot_new = eint_new + KE_new;

			state_arr(i, j, k, px_idx) = px_new;
			state_arr(i, j, k, py_idx) = py_new;
			state_arr(i, j, k, pz_idx) = pz_new;
			state_arr(i, j, k, eint_idx) = eint_new;
			state_arr(i, j, k, etot_idx) = etot_new;
		});
	}
	amrex::Gpu::streamSynchronize();
}

/// @brief Convenes a single half-step of cosmology (for Strang splitting)
/// @return The new scale factor reached
template <typename problem_t>
auto applyCosmologyHalfStep(amrex::MultiFab &state, amrex::Real a_begin, amrex::Real dt_half, CosmologyParams const &cosmo) -> amrex::Real
{
	const amrex::Real a_end = evolveScaleFactor(a_begin, dt_half, cosmo);
	applyCosmologicalSourceTerms<problem_t>(state, a_begin, a_end);
	return a_end;
}

/// @brief Print information about the cosmological model
inline void printCosmologyInfo(CosmologyParams const &cosmo)
{
	// Convert H0 from s^-1 to km/s/Mpc
	const amrex::Real H0_km_s_Mpc = cosmo.H0 * (C::parsec * 1.0e6) / 1.0e5;
	amrex::Print() << "\nCosmological parameters initialized:\n";
	amrex::Print() << "  H0 = " << H0_km_s_Mpc << " km/s/Mpc\n";
	amrex::Print() << "  Omega_m = " << cosmo.Omega_m << "\n";
	amrex::Print() << "  Omega_r = " << cosmo.Omega_r << "\n";
	amrex::Print() << "  Omega_L = " << cosmo.Omega_L << "\n";
	amrex::Print() << "  Current a = 1.0\n\n"; // Will be updated during simulation
}

} // namespace quokka::cosmology

#endif // COSMOLOGY_HPP_
