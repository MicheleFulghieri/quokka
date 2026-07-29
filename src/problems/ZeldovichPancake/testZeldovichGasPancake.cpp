#include "QuokkaSimulation.hpp"
#include "cosmology/Cosmology.hpp"
#include "hydro/hydro_system.hpp"
#include "physics_info.hpp"
#include <cmath>

struct ZeldovichGas {
};

template <> struct quokka::EOS_Traits<ZeldovichGas> {
	static constexpr double gamma = 5.0 / 3.0;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<ZeldovichGas> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_cosmology_enabled = true;
	static constexpr bool is_self_gravity_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = 0;
	static constexpr bool is_dust_enabled = false;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;

	// Cosmology parameters: Einstein-de Sitter universe (flat, matter-only)
	static constexpr double omega_m = 1.0;
	static constexpr double omega_r = 0.0;
	static constexpr double omega_lambda = 0.0;
	static constexpr double hubble_constant = 0.7;	   // h = 0.7 (H0 = 70 km/s/Mpc)
	static constexpr double a_init = 0.02;		       // start at z_init = 49
	static constexpr double cosmology_dt_limit = 0.01; // max delta_a / a per step
};

template <> void QuokkaSimulation<ZeldovichGas>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;      // set of the indices of the grid patch (e.g. from 0 to 31 in x, y, z)
	const amrex::Array4<double> &state_cc = grid_elem.array_;  // Array4 is a pointer to the data

	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &prob_lo = grid_elem.prob_lo_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &prob_hi = grid_elem.prob_hi_;

	const amrex::Real L = prob_hi[0] - prob_lo[0];
	const amrex::Real a_init = PhysicsTraits<ZeldovichGas>::a_init;
	const amrex::Real z_init = 1.0 / a_init - 1.0;
	const amrex::Real z_collapse = 1.0; // collapse at z=1 (a=0.5)
	const amrex::Real a_collapse = 1.0 / (1.0 + z_collapse);

	const amrex::Real T_init = 100.0; // low temperature

	// H(a) for EdS: H(a) = H0 * a^(-3/2)
	const amrex::Real Mpc_to_cm = 3.08567758e24;
	const amrex::Real h = PhysicsTraits<ZeldovichGas>::hubble_constant;
	const amrex::Real H0 = (h * 100.0 * 1e5) / Mpc_to_cm;
	const amrex::Real H_init = H0 * std::pow(a_init, -1.5);
	const amrex::Real G = PhysicsTraits<ZeldovichGas>::gravitational_constant;
	const amrex::Real rho_crit_0 = 3.0 * H0 * H0 / (8.0 * M_PI * G);
	const amrex::Real rho_mean = rho_crit_0; // comoving mean density for EdS

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + 0.5) * dx[0];

		// Zel'dovich pancake initial conditions (1D, x-direction)
		// In the Zel'dovich approximation, linear perturbation theory gives:
		//   delta(q) = -amplitude * cos(k*q)   (density contrast)
		//   rho(q)   = rho_mean / (1 - delta)  (comoving density)
		// where amplitude = a_init / a_collapse < 1  =>  delta is small initially.
		//
		// Growing mode peculiar velocity in EdS (f=1, D=a):
		//   v_pec = a(t) * H(a) * (amplitude / k) * sin(k*q)
		const amrex::Real k_wave = 2.0 * M_PI / L;
		const amrex::Real amplitude = a_init / a_collapse;

		const amrex::Real delta = -amplitude * std::cos(k_wave * x);
		const amrex::Real rho = rho_mean / (1.0 - delta);
		const amrex::Real v = a_init * H_init * (amplitude / k_wave) * std::sin(k_wave * x);

		const amrex::Real gamma = quokka::EOS_Traits<ZeldovichGas>::gamma;
		const amrex::Real eint  = (rho * C::k_B * T_init) / (C::m_u * (gamma - 1.0));

		state_cc(i, j, k, HydroSystem<ZeldovichGas>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ZeldovichGas>::x1Momentum_index) = rho * v;
		state_cc(i, j, k, HydroSystem<ZeldovichGas>::x2Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<ZeldovichGas>::x3Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<ZeldovichGas>::internalEnergy_index) = eint;
		state_cc(i, j, k, HydroSystem<ZeldovichGas>::energy_index) = eint + 0.5 * rho * v * v;
	});
}

template <>
void QuokkaSimulation<ZeldovichGas>::computeReferenceSolution(amrex::MultiFab & /*ref*/, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*dx*/,
								  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_lo*/)
{
}

auto problem_main() -> int
{
	QuokkaSimulation<ZeldovichGas> sim;

	const amrex::Real z_collapse = 1.0;
	const amrex::Real a_collapse = 1.0 / (1.0 + z_collapse);
	const amrex::Real a_init = PhysicsTraits<ZeldovichGas>::a_init;

	// Einstein-de Sitter age: t(a) = (2/3) * (1/H0) * a^(3/2)
	const amrex::Real Mpc_to_cm = 3.08567758e24;
	const amrex::Real h = PhysicsTraits<ZeldovichGas>::hubble_constant;
	const amrex::Real H0 = (h * 100.0 * 1e5) / Mpc_to_cm;
	const amrex::Real t_init = (2.0 / 3.0) * (1.0 / H0) * std::pow(a_init, 1.5);
	const amrex::Real t_collapse = (2.0 / 3.0) * (1.0 / H0) * std::pow(a_collapse, 1.5);

	sim.stopTime_ = t_collapse - t_init;
	sim.maxTimesteps_ = 2000;
	sim.cflNumber_ = 0.3;

	sim.setInitialConditions();
	sim.evolve();

	// Check if the density perturbation has grown
	const amrex::Real rho_min = sim.state_new_cc_[0].min(HydroSystem<ZeldovichGas>::density_index);
	const amrex::Real rho_max = sim.state_new_cc_[0].max(HydroSystem<ZeldovichGas>::density_index);

	amrex::Print() << "\nZel'dovich Pancake Results:\n";
	amrex::Print() << "  Final a = " << sim.a_now_ << " (expected " << a_collapse << ")\n";
	amrex::Print() << "  rho_max / rho_min = " << rho_max / rho_min << "\n";

	int status = 0;
	if (rho_max / rho_min < 10.0) { // Should be very high at collapse
		status = 1;
	}
	return status;
}
