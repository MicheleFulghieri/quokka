#include "QuokkaSimulation.hpp"
#include "cosmology/Cosmology.hpp"
#include "hydro/hydro_system.hpp"
#include "physics_info.hpp"
#include <cmath>

struct ExpansionProblem {};

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
	static constexpr double hubble_constant = 0.7; // h = 0.7 (H0 = 70 km/s/Mpc)
	static constexpr double a_init = 1.0;	       // start at z=0
	static constexpr double cosmology_dt_limit = 1e-4; // very small for accuracy
};

template <> void QuokkaSimulation<ExpansionProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double rho = 1.0e-30; // low density
		const double P = 1.0e-10;  // low pressure
		const double gamma = quokka::EOS_Traits<ExpansionProblem>::gamma;

		state_cc(i, j, k, HydroSystem<ExpansionProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ExpansionProblem>::x1Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<ExpansionProblem>::x2Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<ExpansionProblem>::x3Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<ExpansionProblem>::internalEnergy_index) = P / (gamma - 1.0);
		state_cc(i, j, k, HydroSystem<ExpansionProblem>::energy_index) = P / (gamma - 1.0);
	});
}

template <> void QuokkaSimulation<ExpansionProblem>::computeReferenceSolution(amrex::MultiFab & /*ref*/, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*dx*/,
								  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_lo*/)
{
	// No reference solution needed here, we'll check manually in problem_main
}

auto problem_main() -> int
{
	QuokkaSimulation<ExpansionProblem> sim;

	// Set simulation parameters
	const double yr_to_s = 3.15576e7;
	sim.stopTime_ = 1.0e8 * yr_to_s; // 100 Myr
	sim.maxTimesteps_ = 1000;
	sim.cflNumber_ = 0.3;

	// Initial values
	sim.setInitialConditions();
	const amrex::Real rho0 = sim.state_new_cc_[0].sum(HydroSystem<ExpansionProblem>::density_index) / sim.boxArray(0).numPts();
	const amrex::Real e0 = sim.state_new_cc_[0].sum(HydroSystem<ExpansionProblem>::internalEnergy_index) / sim.boxArray(0).numPts();
	const amrex::Real a0 = 1.0;

	// Evolve
	sim.evolve();

	// Final values
	const amrex::Real rho_f = sim.state_new_cc_[0].sum(HydroSystem<ExpansionProblem>::density_index) / sim.boxArray(0).numPts();
	const amrex::Real e_f = sim.state_new_cc_[0].sum(HydroSystem<ExpansionProblem>::internalEnergy_index) / sim.boxArray(0).numPts();
	const amrex::Real a_f = sim.a_now_;
	const amrex::Real gamma = quokka::EOS_Traits<ExpansionProblem>::gamma;

	// Verify density (should be constant in comoving coords)
	const double rho_err = std::abs(rho_f - rho0) / rho0;
	
	// Verify internal energy scaling: e_f = e0 * (a0/a_f)^(3*(gamma-1))
	const amrex::Real e_expected = e0 * std::pow(a0 / a_f, 3.0 * (gamma - 1.0));
	const double e_err = std::abs(e_f - e_expected) / e_expected;

	amrex::Print() << "\nExpansion Test Results:\n";
	amrex::Print() << "  Final a = " << a_f << "\n";
	amrex::Print() << "  Density error = " << rho_err << " (expected 0)\n";
	amrex::Print() << "  Energy error = " << e_err << " (expected 0)\n";

	int status = 0;
	if (rho_err > 1e-12 || e_err > 1e-4) {
		status = 1;
	}
	return status;
}
