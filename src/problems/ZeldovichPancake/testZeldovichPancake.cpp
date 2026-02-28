#include "QuokkaSimulation.hpp"
#include "cosmology/Cosmology.hpp"
#include "hydro/hydro_system.hpp"
#include "physics_info.hpp"
#include <cmath>

struct ZeldovichProblem {};

template <> struct quokka::EOS_Traits<ZeldovichProblem> {
	static constexpr double gamma = 5.0 / 3.0;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<ZeldovichProblem> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_cosmology_enabled = true;
	static constexpr bool is_self_gravity_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = 0;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nGroups = 1;
	static constexpr int nDustGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;

	// Cosmology parameters (Einstein-de Sitter)
	static constexpr double omega_m = 1.0;
	static constexpr double omega_r = 0.0;
	static constexpr double omega_lambda = 0.0;
	static constexpr double hubble_constant = 0.7; // h = 0.7
	static constexpr double a_init = 0.02;         // start early
	static constexpr double cosmology_dt_limit = 0.01;
};

template <> void QuokkaSimulation<ZeldovichProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &prob_lo = grid_elem.prob_lo_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &prob_hi = grid_elem.prob_hi_;

	const double L = prob_hi[0] - prob_lo[0];
	const double a_init = PhysicsTraits<ZeldovichProblem>::a_init;
	const double z_init = 1.0 / a_init - 1.0;
	const double z_collapse = 1.0; // collapse at z=1 (a=0.5)
	const double a_collapse = 1.0 / (1.0 + z_collapse);
	
	const double T_init = 100.0;     // low temperature

	// H(a) for EdS: H(a) = H0 * a^(-3/2)
	const double Mpc_to_cm = 3.08567758e24;
	const double h = PhysicsTraits<ZeldovichProblem>::hubble_constant;
	const double H0 = (h * 100.0 * 1e5) / Mpc_to_cm;
	const double H_init = H0 * std::pow(a_init, -1.5);
	const double G = PhysicsTraits<ZeldovichProblem>::gravitational_constant;
	const double rho_crit_0 = 3.0 * H0 * H0 / (8.0 * M_PI * G);
	const double rho_mean = rho_crit_0; // comoving mean density for EdS

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + 0.5) * dx[0];
		
		// Zel'dovich ICs
		const double k_wave = 2.0 * M_PI / L;
		const double amplitude = a_init / a_collapse;
		
		const double delta = -amplitude * std::cos(k_wave * x);
		const double rho = rho_mean / (1.0 - delta);
		// v_phys = a^2 * H(a) * psi = a^2 * H(a) * (amplitude/k) * sin(kx)
		const double v = -H_init * (a_init * a_init) * (amplitude / k_wave) * std::sin(k_wave * x);

		const double gamma = quokka::EOS_Traits<ZeldovichProblem>::gamma;
		const double eint = (rho * C::k_B * T_init) / (C::m_u * (gamma - 1.0));

		state_cc(i, j, k, HydroSystem<ZeldovichProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ZeldovichProblem>::x1Momentum_index) = rho * v;
		state_cc(i, j, k, HydroSystem<ZeldovichProblem>::x2Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<ZeldovichProblem>::x3Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<ZeldovichProblem>::internalEnergy_index) = eint;
		state_cc(i, j, k, HydroSystem<ZeldovichProblem>::energy_index) = eint + 0.5 * rho * v * v;
	});
}

template <> void QuokkaSimulation<ZeldovichProblem>::computeReferenceSolution(amrex::MultiFab & /*ref*/, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*dx*/,
								  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_lo*/)
{
}

auto problem_main() -> int
{
	QuokkaSimulation<ZeldovichProblem> sim;

	const double z_collapse = 1.0;
	const double a_collapse = 1.0 / (1.0 + z_collapse);
	const double a_init = PhysicsTraits<ZeldovichProblem>::a_init;
	
	// Einstein-de Sitter age: t(a) = (2/3) * (1/H0) * a^(3/2)
	const double Mpc_to_cm = 3.08567758e24;
	const double h = PhysicsTraits<ZeldovichProblem>::hubble_constant;
	const double H0 = (h * 100.0 * 1e5) / Mpc_to_cm;
	const double t_init = (2.0/3.0) * (1.0/H0) * std::pow(a_init, 1.5);
	const double t_collapse = (2.0/3.0) * (1.0/H0) * std::pow(a_collapse, 1.5);

	sim.stopTime_ = t_collapse - t_init;
	sim.maxTimesteps_ = 2000;
	sim.cflNumber_ = 0.3;

	sim.setInitialConditions();
	sim.evolve();

	// Check if the density perturbation has grown
	const amrex::Real rho_min = sim.state_new_cc_[0].min(HydroSystem<ZeldovichProblem>::density_index);
	const amrex::Real rho_max = sim.state_new_cc_[0].max(HydroSystem<ZeldovichProblem>::density_index);
	
	amrex::Print() << "\nZel'dovich Pancake Results:\n";
	amrex::Print() << "  Final a = " << sim.a_now_ << " (expected " << a_collapse << ")\n";
	amrex::Print() << "  rho_max / rho_min = " << rho_max / rho_min << "\n";

	int status = 0;
	if (rho_max / rho_min < 10.0) { // Should be very high at collapse
		status = 1;
	}
	return status;
}
