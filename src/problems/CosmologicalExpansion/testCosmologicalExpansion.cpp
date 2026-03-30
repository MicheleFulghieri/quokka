//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testCosmologicalExpansion.cpp
/// \brief Defines a test problem for a cosmological expansion.
///

#include "QuokkaSimulation.hpp"
#include "cosmology/Cosmology.hpp"
#include "hydro/hydro_system.hpp"
#include "physics_info.hpp"
#include <cmath>

struct ExpansionProblem {
	static constexpr double rho0_default = 1.0e-30; // low density
	static constexpr double P0_default = 1.0e-10;	// low pressure
};

// energy evolution for the analytical solution
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto get_analytic_energy(amrex::Real e_init, amrex::Real a0, amrex::Real a_fin, double gamma) -> amrex::Real
{
	return e_init * std::pow(a0 / a_fin, 3.0 * (gamma - 1.0));
}

template <> struct quokka::EOS_Traits<ExpansionProblem> {
	static constexpr double gamma = 5.0 / 3.0;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<ExpansionProblem> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_cosmology_enabled = true;
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = 0;
	static constexpr bool is_dust_enabled = false;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;

	// Cosmology parameters: flat matter-only EdS universe
	static constexpr double omega_m = 1.0;
	static constexpr double omega_r = 0.0;
	static constexpr double omega_lambda = 0.0;
	static constexpr double hubble_constant = 0.7;	   // h = 0.7 (H0 = 70 km/s/Mpc)
	static constexpr double a_init = 1.0;		   // start at z=0
	static constexpr double cosmology_dt_limit = 1e-4; // very small for accuracy
};

// Specialize the empty function of QuokkaSimualation.hpp to this problem
template <> void QuokkaSimulation<ExpansionProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// Initial conditions: default of the problem
	const double gamma = quokka::EOS_Traits<ExpansionProblem>::gamma;
	amrex::Real rho = ExpansionProblem::rho0_default;
	amrex::Real P = ExpansionProblem::P0_default;

	// Initial conditions: precedence to user-entered values (override if present in the .in file)
	amrex::ParmParse pp("problem");
	pp.query("rho0", rho);
	pp.query("P0", P);

	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<ExpansionProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ExpansionProblem>::x1Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<ExpansionProblem>::x2Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<ExpansionProblem>::x3Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<ExpansionProblem>::internalEnergy_index) = P / (gamma - 1.0);
		state_cc(i, j, k, HydroSystem<ExpansionProblem>::energy_index) = P / (gamma - 1.0);
	});
}

// Analytical solution for the expansion test: density should remain constant in comoving coordinates
// and internal energy should scale as a^(-3*(gamma-1))
// Saved in plotfile outputs and useful for direct comparison. The numerical comparison is instead done
// in the problem_main
template <>
void QuokkaSimulation<ExpansionProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*dx*/,
								  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_lo*/)
// amrex::MultiFab &ref -> pass by ref the empty collection of patches (grids) distributed on the CPU or GPU, the MultiFab object ref
// commented the dimension of the cells (dx) and the coordinate origin (prob_lo), since the problem is uniform
{
	// Physical parameters
	const amrex::Real gamma = quokka::EOS_Traits<ExpansionProblem>::gamma;
	amrex::Real a_init = Physics_Traits<ExpansionProblem>::a_init;
	amrex::ParmParse pp_cosmo("cosmology");
	pp_cosmo.query("a_init", a_init);
	amrex::Real a_now = this->a_now_; // current scale factor (member of the class QuokkaSimulation), update by solving the Friedmann equation

	// Initial conditions: default if not otherwise declared by the user in the .in file
	amrex::Real rho = ExpansionProblem::rho0_default;
	amrex::Real P = ExpansionProblem::P0_default;
	amrex::ParmParse pp("problem");
	pp.query("rho0", rho);
	pp.query("P0", P);

	// Initial energy and analytic solutions
	const amrex::Real e0 = P / (gamma - 1);
	const amrex::Real e_sol = get_analytic_energy(e0, a_init, a_now, gamma);
	const amrex::Real rho_sol = rho; // comoving density doesn't change

	// Get the pointers (Array4) of the grid collection MultiFab ref
	auto const &ref_arrays = ref.arrays();

	// Kernel GPU to fill the reference solution captured by [=]
	amrex::ParallelFor(ref, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
		ref_arrays[box_no](i, j, k, HydroSystem<ExpansionProblem>::density_index) = rho;
		ref_arrays[box_no](i, j, k, HydroSystem<ExpansionProblem>::x1Momentum_index) = 0;
		ref_arrays[box_no](i, j, k, HydroSystem<ExpansionProblem>::x2Momentum_index) = 0;
		ref_arrays[box_no](i, j, k, HydroSystem<ExpansionProblem>::x3Momentum_index) = 0;
		// v = 0, them total energy = internal energy
		ref_arrays[box_no](i, j, k, HydroSystem<ExpansionProblem>::energy_index) = e_sol;
		ref_arrays[box_no](i, j, k, HydroSystem<ExpansionProblem>::internalEnergy_index) = e_sol;
	});

	// GPU synchronization: CPU waits until the GPU has written all the values in the MultiFabs
	amrex::Gpu::streamSynchronize();
}

auto problem_main() -> int
{
	QuokkaSimulation<ExpansionProblem> sim;

	sim.readParameters();

	// Set simulation parameters
	const double yr_to_s = 3.15576e7;
	sim.stopTime_ = 1.0e8 * yr_to_s; // 100 Myr
	sim.maxTimesteps_ = 1000;
	sim.cflNumber_ = 0.3;
	amrex::Real a_f = sim.a_now_; // add to this scope

	amrex::ParmParse pp_amr("amr");
	pp_amr.query("max_timesteps", sim.maxTimesteps_);
	if (pp_amr.query("stop_time", sim.stopTime_)) { // if to avoid double moltiplication for yr_to_s
		sim.stopTime_ *= yr_to_s;
	}
	amrex::ParmParse pp_quokka("quokka");
	pp_quokka.query("cfl", sim.cflNumber_);

	// Initial values
	// sim.state_new_cc_[0]: state_new_cc_ is the Quokka amrex::Vector<amrex::MultiFab> variable which stores the HD data a the next time, [0] is the (base)
	// level of the grid. It is summed in order to calculate the average value.
	// / sim.boxArray(0).numPts(): the denominator of the mean value, boxArray(0) defines the geometry of level 0 of AMR. numPts() returns the total number
	// of cells (points) in the simulation. OBS: in principle the average is not necessary, since density and energy are uniform, but so it is possible to
	// compansate for the singule machine-precision fluctuation of each cell
	sim.setInitialConditions();
	const amrex::Real rho0 = sim.state_new_cc_[0].sum(HydroSystem<ExpansionProblem>::density_index) / sim.boxArray(0).numPts();
	const amrex::Real e0 = sim.state_new_cc_[0].sum(HydroSystem<ExpansionProblem>::internalEnergy_index) / sim.boxArray(0).numPts();

	// Scale factor today: default if not otherwise declared
	amrex::Real a0 = Physics_Traits<ExpansionProblem>::a_init;
	amrex::ParmParse pp_cosmo("cosmology");
	pp_cosmo.query("a_init", a0);

	// Evolve
	sim.evolve();

	sim.WritePlotFile();

	// Error calculation : L1 and Linf norms
	auto const &mf_sim = sim.state_new_cc_[0]; // the MultiFab of the simulation at the final time
	// Allocation of the new MultiFab mf_ref for the analytical solution, copying the spatial structure and division between the simulation processors
	// (boxArray(0) and DistributionMap(0)), the same number of components ( mf_sim.nComp()), but 0 ghost cells
	amrex::MultiFab mf_ref(mf_sim.boxArray(), mf_sim.DistributionMap(), mf_sim.nComp(),
			       0); // empty MultiFab with the same geometry and distribution as the simulation, but with 0 ghost cells

	// Calculation of the current analytical solution
	// Geom(0) passes geometric information (cell size and origin coordinates)
	sim.computeReferenceSolution(mf_ref, sim.Geom(0).CellSizeArray(),
				     sim.Geom(0).ProbLoArray()); // use the method computeReferenceSolution of QuokkaSimualtion, now specialized to this problem

	// Error calculation: simulation - reference
	amrex::MultiFab mf_err(mf_sim.boxArray(), mf_sim.DistributionMap(), mf_sim.nComp(), 0); // allocate a new MultiFab to store the error of each cell
	amrex::MultiFab::Copy(mf_err, mf_sim, 0, 0, mf_sim.nComp(), 0);				// copy all the simulationd data (mf_sim) in mf_err
	amrex::MultiFab::Subtract(mf_err, mf_ref, 0, 0, mf_sim.nComp(), 0);			// cell error = sim cell - ref cell

	// Density: calculation of the norms
	const int rho_idx = HydroSystem<ExpansionProblem>::density_index; // retrive the index of the density
	const amrex::Real rho_norm = mf_ref.norm1(rho_idx);		  // norm1 sums the absolute values of each cell

	// L1 error: mean absolute error
	amrex::Real L1_rho = mf_err.norm1(rho_idx) / rho_norm; // mean relative error
	// L2 norm: mean square error
	amrex::Real L2_rho = mf_err.norm2(rho_idx) / mf_ref.norm2(rho_idx); // mean square relative error
	// Linf norm: max absolute error
	amrex::Real Linf_rho = mf_err.norminf(rho_idx) / mf_ref.norminf(rho_idx); // norminf returns the highest value in the cell

	// Internal energy: calculation of the norm
	const int eint_idx = HydroSystem<ExpansionProblem>::internalEnergy_index;
	const amrex::Real eint_norm = mf_ref.norm1(eint_idx);

	// L1 norm
	amrex::Real L1_eint = mf_err.norm1(eint_idx) / eint_norm;
	// L2 norm
	amrex::Real L2_eint = mf_err.norm2(eint_idx) / mf_ref.norm2(eint_idx);
	// Linf norm
	amrex::Real Linf_eint = mf_err.norminf(eint_idx) / mf_ref.norminf(eint_idx);

	// Final print
	amrex::Print() << "\nVerification Norms \n";
	amrex::Print() << "  Density: L1 = " << L1_rho << " | L2 = " << L2_rho << " | Linf = " << Linf_rho << "\n";
	amrex::Print() << "  Energy : L1 = " << L1_eint << " | L2 = " << L2_eint << " | Linf = " << Linf_eint << "\n";

	amrex::Print() << "\nExpansion Test Results:\n";
	amrex::Print() << "  Final a = " << a_f << "\n";

	// Test status
	int status = 0; // success

	// Tolerances
	const amrex::Real tol_rho = 1e-12; // density is constant, so the only error is floating point
	const amrex::Real tol_eint = 1e-4; // numerical integration involved, so more permissive treshold

	// L1 norm as a reference for the error
	if (L1_rho > tol_rho || L1_eint > tol_eint) {
		// amrex::Print() << "TEST FAILED: Error exceeds tolerances!\n";
		status = 1;
	} else {
		amrex::Print() << "TEST PASSED.\n";
	}
	return status;
}

// Controllare se amrex::norm2 estrae gia la radice, altrimenti correggere con:
// amrex::Real L2_rho = std::sqrt(mf_err.norm2(rho_idx)) / std::sqrt(mf_ref.norm2(rho_idx));
// amrex::Real L2_eint = std::sqrt(mf_err.norm2(eint_idx)) / std::sqrt(mf_ref.norm2(eint_idx));

// Introdurre opzioni scelta sia da pp per costante di hubble e omega_, lasciando EdS come default
// Forse non è necessario, perché in QuokkaSimulation.hpp già imposta questa gerarchia di precedenze.
// Provare con un test con diversi input .in
