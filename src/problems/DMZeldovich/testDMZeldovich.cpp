//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testCosmologicalDarkMatter.cpp
/// \brief This problem tests the interplay between hydro, CIC particles, gravity and
/// cosmological expansion. A caustic formation in the center of the domain with
/// a density and temperature peak is expected (Zel’dovich, 1970).
///

#include "QuokkaSimulation.hpp"
#include "particles/particle_types.hpp"
#include "cosmology/Cosmology.hpp"
#include "math/root_finding.hpp"

#include <AMReX_Math.H>
#include <AMReX_ParticleMesh.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>


struct DMZeldovich {
};

template <> struct quokka::EOS_Traits<DMZeldovich> {
	static constexpr amrex::Real gamma = 5.0 / 3.0;
	static constexpr amrex::Real mean_molecular_weight = C::m_u;
};

template <> struct Particle_Traits<DMZeldovich>{  // CIC from particle_types.hpp
    static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC;
};

template <> struct Physics_Traits<DMZeldovich> {
    static constexpr bool is_hydro_enabled        = true;
	static constexpr bool is_cosmology_enabled    = true;
	static constexpr bool is_self_gravity_enabled = true;
	static constexpr bool is_radiation_enabled    = false;
	static constexpr bool is_mhd_enabled          = false;
	static constexpr int numMassScalars           = 0;
	static constexpr int numPassiveScalars        = 0;
	static constexpr bool is_dust_enabled         = false;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;

// Cosmology parameters
	static constexpr amrex::Real omega_m = 1.0;             // EdS universe
	static constexpr amrex::Real omega_r = 0.0;
	static constexpr amrex::Real omega_lambda = 0.0;
	static constexpr amrex::Real omega_b = 0.01;            // 99% DMd
	static constexpr amrex::Real omega_dm = 0.99;
	static constexpr amrex::Real hubble_constant = 0.7;	   // h = 0.7 (H0 = 70 km/s/Mpc)
	static constexpr amrex::Real a_init = 0.01;		       // start at z = 99
	static constexpr amrex::Real cosmology_dt_limit = 0.01; // according to the default
};


template <> void QuokkaSimulation<DMZeldovich>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {

	amrex::Real a_init = PhysicsTraits<DMZeldovich>::a_init;
    amrex::Real a_collapse = 0.5;
    amrex::ParmParse pp_cosmo("cosmology");
    pp_cosmo.query("a_init", a_init);
    pp_cosmo.query("a_collapse", a_collapse);

	const amrex::Real G = PhysicsTraits<DMZeldovich>::gravitational_constant;
	const amrex::Real h = PhysicsTraits<DMZeldovich>::hubble_constant;
	const amrex::Real Mpc_to_cm = C::parsec * 1.0e6; 
	const amrex::Real H0 = (h * 100.0 * 1e5) / Mpc_to_cm;    // Hubble parameter today (s^-1)
	const amrex::Real H_init = H0 * std::pow(a_init, -1.5);  // initial Hubble paramter for EdS from H0
	const amrex::Real rho_crit_0 = 3.0 * H0 * H0 / (8.0 * amrex::Math::pi<amrex::Real>() * G);
	const amrex::Real rho_mean = rho_crit_0;                 // comoving mean density = critical density (EdS flat universe)

	// Thermodynamical status
	const amrex::Real mu = quokka::EOS_Traits<DMZeldovich>::mean_molecular_weight;
	const amrex::Real gamma = quokka::EOS_Traits<DMZeldovich>::gamma;
	const amrex::Real rho_gas = Physics_Traits<DMZeldovich>::omega_b * rho_mean;  // according to the budget
    const amrex::Real T_init = 100.0;      // [K] start with cold gas
	const amrex::Real P_mean = (rho_gas * C::k_B * T_init) / mu;  // ideal gas eos P = rho * Kb * T / mu
	 
	// Gas perturbation
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;       
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo  = grid_elem.prob_lo_;  
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi  = grid_elem.prob_hi_;  
	const amrex::Real L_box = prob_hi[0] - prob_lo[0];
	const amrex::Real k_wave = 2.0 * M_PI / L_box;
	const amrex::Real amplitude  = a_init / a_collapse; 

    const amrex::Box &indexRange = grid_elem.indexRange_;           // set of the indices of the grid patch 
    const amrex::Array4<amrex::Real> &state_cc = grid_elem.array_;  // Array4 is a pointer to the data

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const amrex::Real x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];  // x cell center
		const amrex::Real x_w = x - L_box * std::floor(x / L_box);  // wrap Eulerian position into [0, L_box): root finder fails for ghost cells

		// Function f(q) = q - (A/k)*sin(k*q) - x
		auto f_zel = [=](amrex::Real q) {   
			return q - (amplitude / k_wave) * std::sin(k_wave * q) - x_w;
		};

		// Zeros of f(q) via Quokka's root solver toms748_solve
		const amrex::Real fa = f_zel(0.0);
		const amrex::Real fb = f_zel(L_box);
		int max_iter = 50;
		const auto [qa, qb] = quokka::math::toms748_solve(  // return the structure binding of the interval enclosing the solution
		    f_zel, amrex::Real(0.0), L_box, fa, fb,
			quokka::math::eps_tolerance<amrex::Real>(static_cast<unsigned int>(30)), max_iter);  // stop when difference between the intterval extremes is less than 30 times the machine epsilon
		const amrex::Real q = 0.5 * (qa + qb);  // lagrangian coordinate solution as interval midpoint
        // Shift q to centre the collapse (q_center = q - L_box/2)
        const amrex::Real q_center = q - 0.5 * L_box;
        // Initialize hydro using shifted coordinate
        const amrex::Real rho = rho_gas / (1.0 - amplitude * std::cos(k_wave * q_center)); // collapse at domain centre
        const amrex::Real v = - a_init * H_init * (amplitude / k_wave) * std::sin(k_wave * q_center);
        const amrex::Real P_local = P_mean * std::pow(rho / rho_gas, gamma);
        const amrex::Real eint = quokka::EOS<DMZeldovich>::ComputeEintFromPres(rho, P_local);

        state_cc(i, j, k, HydroSystem<DMZeldovich>::density_index)    = rho; 
        state_cc(i, j, k, HydroSystem<DMZeldovich>::x1Momentum_index) = rho * v;
        state_cc(i, j, k, HydroSystem<DMZeldovich>::x2Momentum_index) = 0;
        state_cc(i, j, k, HydroSystem<DMZeldovich>::x3Momentum_index) = 0;
        state_cc(i, j, k, HydroSystem<DMZeldovich>::internalEnergy_index) = eint;
        state_cc(i, j, k, HydroSystem<DMZeldovich>::energy_index) = eint + 0.5 * rho * (v * v);
    });
}


template <> void QuokkaSimulation<DMZeldovich>::createInitialCICParticles() {
	amrex::Real a_init = PhysicsTraits<DMZeldovich>::a_init;
	amrex::Real a_collapse = 0.5;
	amrex::ParmParse pp_cosmo("cosmology");
	pp_cosmo.query("a_init", a_init);
	pp_cosmo.query("a_collapse", a_collapse);

	const amrex::Real G = PhysicsTraits<DMZeldovich>::gravitational_constant;
	const amrex::Real h = PhysicsTraits<DMZeldovich>::hubble_constant;
	const amrex::Real Mpc_to_cm = C::parsec * 1.0e6; 
	const amrex::Real H0 = (h * 100.0 * 1e5) / Mpc_to_cm;    // Hubble parameter today (s^-1)
	const amrex::Real H_init = H0 * std::pow(a_init, -1.5);  // initial Hubble paramter for EdS from H0
	const amrex::Real rho_crit_0 = 3.0 * H0 * H0 / (8.0 * amrex::Math::pi<amrex::Real>() * G);
	const amrex::Real rho_mean = rho_crit_0;                 // comoving mean density = critical density (EdS flat universe)
	
	const int lev = 0;
	auto &pc = *CICParticles;                       // take by ref the content of the unique_ptr CICParticles 
	const amrex::Geometry &geom = this->geom[lev];
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = geom.ProbHiArray();
	const amrex::Real Lx = prob_hi[0] - prob_lo[0];
	const amrex::Real Ly = prob_hi[1] - prob_lo[1];
	const amrex::Real Lz = prob_hi[2] - prob_lo[2];

	int nparx = 32;
	amrex::ParmParse pp_part("particles");
	pp_part.query("npar_x", nparx);
	int npary = nparx;
	int nparz = nparx;
	pp_part.query("npar_y", npary);
	pp_part.query("npar_z", nparz);
	const amrex::Long npartot = static_cast<amrex::Long>(nparx) * npary * nparz;
	
	// Particle masses and distances
	const amrex::Real tot_rho_dm  = Physics_Traits<DMZeldovich>::omega_dm * rho_mean;  // according to the budget
	const amrex::Real tot_mass_dm = tot_rho_dm * Lx * Ly * Lz;
	const amrex::Real part_mass   = tot_mass_dm / npartot;
	const amrex::Real dpartx      = Lx / nparx;
	const amrex::Real dparty      = Ly / npary;
	const amrex::Real dpartz      = Lz / nparz;

	const amrex::Real amplitude   = a_init / a_collapse;   // force collapse at a_collapse
	const amrex::Real k_wave      = (2.0 * amrex::Math::pi<amrex::Real>()) / Lx;

	amrex::Print() << "DM particles initialization along x,y,z: " 
	               << nparx << "x" << npary << "x" << nparz << ")..." << std::endl;

	if (amrex::ParallelDescriptor::IOProcessor()) {
		using ParticleType = quokka::CICParticleContainer::ParticleType;
		
		// We get the first local tile key for processor 0 to temporarily host the particles
		amrex::MFIter mfi(state_new_cc_[lev]);
		if (mfi.isValid()) {
			auto const key = std::make_pair(mfi.index(), mfi.LocalTileIndex());
			auto &particles = pc.GetParticles(lev)[key];
			
			for (int i = 0; i < nparx; ++i) {
				for (int j = 0; j < npary; ++j) {
					for (int k = 0; k < nparz; ++k) {
						amrex::Real qx = prob_lo[0] + (i + 0.5) * dpartx;
						amrex::Real qy = prob_lo[1] + (j + 0.5) * dparty;
						amrex::Real qz = prob_lo[2] + (k + 0.5) * dpartz;

        // Shift qx to centre the collapse (qx_center = qx - L_box/2)
        const amrex::Real qx_center = qx - 0.5 * Lx;
        const amrex::Real displacement = - (amplitude / k_wave) * std::sin(k_wave * qx_center);
        const amrex::Real v_pec        = a_init * H_init * displacement; 

						// Ensure periodic boundary wrapping
						amrex::Real x_perturbed = qx + displacement;
						while (x_perturbed < prob_lo[0]) x_perturbed += Lx;
						while (x_perturbed >= prob_hi[0]) x_perturbed -= Lx;

						amrex::Real y_perturbed = qy;
						while (y_perturbed < prob_lo[1]) y_perturbed += Ly;
						while (y_perturbed >= prob_hi[1]) y_perturbed -= Ly;

						amrex::Real z_perturbed = qz;
						while (z_perturbed < prob_lo[2]) z_perturbed += Lz;
						while (z_perturbed >= prob_hi[2]) z_perturbed -= Lz;

						ParticleType p;  
						p.id()  = ParticleType::NextID();
						p.cpu() = amrex::ParallelDescriptor::MyProc();
						p.pos(0) = x_perturbed;
						p.pos(1) = y_perturbed;
						p.pos(2) = z_perturbed;			
						p.rdata(quokka::CICParticleMassIdx) = part_mass;   
						p.rdata(quokka::CICParticleVxIdx)   = v_pec;		     
						p.rdata(quokka::CICParticleVyIdx)   = 0.0;		     
						p.rdata(quokka::CICParticleVzIdx)   = 0.0;		

						particles.push_back(p);
					}
				}
			}
		} else {
			amrex::Abort("Error: 'IOProcessor' has no local grid assigned to level 0!");
		}
	}
	pc.Redistribute();
	
	amrex::Long total_particles = pc.TotalNumberOfParticles();  
	amrex::Print() << "Initialization of " << total_particles << " DM particles completed.\n" << std::endl;
}

template <> void QuokkaSimulation<DMZeldovich>::computeReferenceSolution(amrex::MultiFab &ref,
                                                                         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
                                                                         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	const amrex::Real a_now = this->a_now_;
	amrex::Real a_init = PhysicsTraits<DMZeldovich>::a_init;
	amrex::Real a_collapse = 0.5;
	amrex::ParmParse pp_cosmo("cosmology");
	pp_cosmo.query("a_init", a_init);
	pp_cosmo.query("a_collapse", a_collapse);

	const amrex::Real L_x = this->geom[0].ProbHi(0) - this->geom[0].ProbLo(0);
	const amrex::Real k_wave = 2.0 * amrex::Math::pi<amrex::Real>() / L_x;
	const amrex::Real A_now = a_now / a_collapse;

	const amrex::Real G = PhysicsTraits<DMZeldovich>::gravitational_constant;
	const amrex::Real h = PhysicsTraits<DMZeldovich>::hubble_constant;
	const amrex::Real Mpc_to_cm = C::parsec * 1.0e6;
	const amrex::Real H0 = (h * 100.0 * 1e5) / Mpc_to_cm;
	const amrex::Real H_now = H0 * std::pow(a_now, -1.5);
	const amrex::Real rho_crit_0 = 3.0 * H0 * H0 / (8.0 * amrex::Math::pi<amrex::Real>() * G);
	const amrex::Real rho_mean = rho_crit_0;
	const amrex::Real rho_gas = PhysicsTraits<DMZeldovich>::omega_b * rho_mean;

	const amrex::Real gamma = quokka::EOS_Traits<DMZeldovich>::gamma;
	const amrex::Real mu = quokka::EOS_Traits<DMZeldovich>::mean_molecular_weight;
	const amrex::Real T_init = 100.0;
	const amrex::Real P_mean = (rho_gas * C::k_B * T_init) / mu;

	auto const &ref_arrays = ref.arrays();

	amrex::ParallelFor(ref, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
		const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
		// Wrap Eulerian position into [0, L_x)
		const amrex::Real x_w = x - L_x * std::floor(x / L_x);

		// Invert x = q - (A/k)*sin(k*q) analytically using Quokka's root solver.
		// f(q) = q - (A/k)*sin(k*q) - x_w.  Before collapse A<1 so f is monotone
		// on [0, L_x]: f(0)=-x_w<=0, f(L_x)=L_x-x_w>=0.
		auto f_zel = [=](amrex::Real q) {
			return q - (A_now / k_wave) * std::sin(k_wave * q) - x_w;
		};
		const amrex::Real fa = f_zel(0.0);
		const amrex::Real fb = f_zel(L_x);
		int max_iter = 50;
		const auto [qa, qb] = quokka::math::toms748_solve(
		    f_zel, amrex::Real(0.0), L_x, fa, fb,
			quokka::math::eps_tolerance<amrex::Real>(static_cast<unsigned int>(30)), max_iter);
		const amrex::Real q = 0.5 * (qa + qb);

		// Analytical Zel'dovich solution at Lagrangian coordinate        // Shift q to centre the analytical solution
        const amrex::Real q_center = q - 0.5 * L_x;
        const amrex::Real rho_sol  = rho_gas / (1.0 - A_now * std::cos(k_wave * q_center));
        const amrex::Real v_sol    = -a_now * H_now * (A_now / k_wave) * std::sin(k_wave * q_center);
        const amrex::Real P_local  = P_mean * std::pow(rho_sol / rho_gas, gamma);
		const amrex::Real eint_sol = P_local / (gamma - 1.0);

		ref_arrays[box_no](i, j, k, HydroSystem<DMZeldovich>::density_index)         = rho_sol;
		ref_arrays[box_no](i, j, k, HydroSystem<DMZeldovich>::x1Momentum_index)      = rho_sol * v_sol;
		ref_arrays[box_no](i, j, k, HydroSystem<DMZeldovich>::x2Momentum_index)      = 0.0;
		ref_arrays[box_no](i, j, k, HydroSystem<DMZeldovich>::x3Momentum_index)      = 0.0;
		ref_arrays[box_no](i, j, k, HydroSystem<DMZeldovich>::internalEnergy_index)   = eint_sol;
		ref_arrays[box_no](i, j, k, HydroSystem<DMZeldovich>::energy_index)           = eint_sol + 0.5 * rho_sol * v_sol * v_sol;
	});
	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<DMZeldovich>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const
{
	if (dname == "gpot") {
		const int ncomp = ncomp_cc_in;
		auto const &phi_arr = phi[lev].const_arrays();
		auto output = mf.arrays();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept { output[bx](i, j, k, ncomp) = phi_arr[bx](i, j, k); });
	}
}

static int regression_status = 0;   // NOLINT
static bool regression_checked = false; // NOLINT

template <> void QuokkaSimulation<DMZeldovich>::computeAfterTimestep()
{
	amrex::Real a_collapse = 0.5;
	amrex::ParmParse pp_cosmo("cosmology");
	pp_cosmo.query("a_collapse", a_collapse);
	const amrex::Real a_target = 0.8 * a_collapse;

	if (!regression_checked && a_now_ >= a_target) {
		amrex::Print() << "\n=== Zel'dovich Regression Check at a = " << a_now_ << " ===\n";

		// --- Gas check: compare numerical state against the analytical Zel'dovich solution ---
		auto const &mf_sim = state_new_cc_[0];
		amrex::MultiFab mf_ref(mf_sim.boxArray(), mf_sim.DistributionMap(), mf_sim.nComp(), 0);
		computeReferenceSolution(mf_ref, Geom(0).CellSizeArray(), Geom(0).ProbLoArray());

		amrex::MultiFab mf_err(mf_sim.boxArray(), mf_sim.DistributionMap(), mf_sim.nComp(), 0);
		amrex::MultiFab::Copy(mf_err, mf_sim, 0, 0, mf_sim.nComp(), 0);
		amrex::MultiFab::Subtract(mf_err, mf_ref, 0, 0, mf_sim.nComp(), 0);

		const int rho_idx  = HydroSystem<DMZeldovich>::density_index;
		const int eint_idx = HydroSystem<DMZeldovich>::internalEnergy_index;
		const int momx_idx = HydroSystem<DMZeldovich>::x1Momentum_index;

		// Relative L1 norm: ||num - ref||_1 / ||ref||_1
		auto get_relative_l1 = [&](int idx) -> amrex::Real {
			const amrex::Real norm_ref = mf_ref.norm1(idx);
			return (norm_ref > 0) ? (mf_err.norm1(idx) / norm_ref) : mf_err.norm1(idx);
		};

		const amrex::Real err_rho  = get_relative_l1(rho_idx);
		const amrex::Real err_eint = get_relative_l1(eint_idx);
		const amrex::Real err_momx = get_relative_l1(momx_idx);

		// --- DM particle check: analytical EdS velocity field on the Lagrangian grid ---
		//
		// In EdS, with perturbation x = q - (A(a)/k)*sin(k*q), the peculiar velocity is:
		//   v(q, a) = -a * H(a) * (A(a)/k) * sin(k*q)
		//
		// Particles were placed on a regular Lagrangian grid q_i = (i+0.5)*dq,
		// so the analytical L1 norm over N_x particles reduces to:
		//   sum_i |v(q_i)| = N_tot * a*H(a)*(A/k) * (1/N_x) * sum_i |sin(k*q_i)|
		//
		// For a uniform grid over one full wavelength:
		//   (1/N_x) * sum_i |sin(2*pi*i/N_x)| --> 2/pi as N_x -> inf  (exact for any even N_x)
		//
		// We use the exact discrete sum to avoid any large-N approximation.

		const amrex::Geometry &geom = this->geom[0];
		const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
		const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = geom.ProbHiArray();
		const amrex::Real Lx = prob_hi[0] - prob_lo[0];
		const amrex::Real k_wave = 2.0 * amrex::Math::pi<amrex::Real>() / Lx;
		const amrex::Real a_now = a_now_;
		const amrex::Real A_now = a_now / a_collapse;

		const amrex::Real h_cosmo = PhysicsTraits<DMZeldovich>::hubble_constant;
		const amrex::Real Mpc_to_cm = C::parsec * 1.0e6;
		const amrex::Real H0 = (h_cosmo * 100.0 * 1.0e5) / Mpc_to_cm;
		const amrex::Real H_now = H0 * std::pow(a_now, -1.5); // EdS: H(a) = H0 * a^{-3/2}
		const amrex::Real v_amplitude = a_now * H_now * (A_now / k_wave); // scalar amplitude

		// Read particle counts from the input (must match createInitialCICParticles)
		int nparx = 32;
		amrex::ParmParse pp_part("particles");
		pp_part.query("npar_x", nparx);
		const amrex::Real dpartx = Lx / nparx;

		// Exact discrete sum: sum_{i=0}^{nparx-1} |sin(k * (i+0.5)*dpartx)|
		// = sum_{i=0}^{nparx-1} |sin(pi*(i+0.5)/nparx)|   (since k*dpartx = 2pi/nparx)
		amrex::Real sum_sin = 0.0;
		for (int i = 0; i < nparx; ++i) {
			const amrex::Real q_i = prob_lo[0] + (i + 0.5) * dpartx;
			sum_sin += std::abs(std::sin(k_wave * q_i));
		}

		// Total analytical L1 of |v_x| over all N_tot particles
		// (y, z directions are uniform and don't affect sin(k*q_x))
		int npary = nparx;
		int nparz = nparx;
		pp_part.query("npar_y", npary);
		pp_part.query("npar_z", nparz);
		const amrex::Long npartot = static_cast<amrex::Long>(nparx) * npary * nparz;
		const amrex::Real analytical_l1_v = v_amplitude * sum_sin * npary * nparz;

		// Numerical L1 of |v_x| summed over all local particles, then MPI-reduced
		const auto [real_data, int_data] =
		    particleRegister_.getParticleDescriptor(quokka::ParticleType::CIC)->getParticleDataAtLevel(0);

		amrex::Real numerical_l1_v = 0.0;
		// real_data layout per particle: [pos_x, pos_y, pos_z, mass, vx, vy, vz]
		// positions occupy indices 0-2; rdata starts at index 3.
		constexpr int vx_offset = 3 + quokka::CICParticleVxIdx; // = 4
		for (size_t i = 0; i < real_data.size(); ++i) {
			numerical_l1_v += std::abs(real_data[i][vx_offset]);
		}
		amrex::ParallelDescriptor::ReduceRealSum(numerical_l1_v);

		const amrex::Real err_part_v = (analytical_l1_v > 0)
		    ? std::abs(numerical_l1_v - analytical_l1_v) / analytical_l1_v
		    : std::abs(numerical_l1_v);

		amrex::Print() << "  Gas density    L1 rel. error: " << err_rho  << "\n"
		               << "  Gas int.energy L1 rel. error: " << err_eint << "\n"
		               << "  Gas momentum x L1 rel. error: " << err_momx << "\n"
		               << "  DM |vx| L1 rel. error:        " << err_part_v
		               << "  (" << npartot << " particles)\n";

		// Tolerances (linear regime at 0.8*a_collapse, grid resolution ~3%)
		const amrex::Real tol_rho  = 0.05; // 5%
		const amrex::Real tol_eint = 0.08; // 8% (adiabatic compression adds higher-order terms)
		const amrex::Real tol_mom  = 0.08; // 8%
		const amrex::Real tol_part = 0.05; // 5%

		if (err_rho > tol_rho || err_eint > tol_eint || err_momx > tol_mom || err_part_v > tol_part) {
			amrex::Print() << "REGRESSION FAILED: errors exceed tolerances.\n";
			regression_status = 1;
		} else {
			amrex::Print() << "REGRESSION PASSED.\n";
			regression_status = 0;
		}
		regression_checked = true;
	}
}

auto problem_main() -> int {
	int status = 0;
	QuokkaSimulation<DMZeldovich> sim;

	sim.readParameters();

	amrex::Real a_init = PhysicsTraits<DMZeldovich>::a_init;
	amrex::Real a_collapse = 0.5;
	amrex::ParmParse pp_cosmo("cosmology");
	pp_cosmo.query("a_init", a_init);
	pp_cosmo.query("a_collapse", a_collapse);

	const amrex::Real G = PhysicsTraits<DMZeldovich>::gravitational_constant;
	const amrex::Real h = PhysicsTraits<DMZeldovich>::hubble_constant;
	const amrex::Real Mpc_to_cm = C::parsec * 1.0e6; 
	const amrex::Real H0 = (h * 100.0 * 1e5) / Mpc_to_cm;

	// Evolve until a_final (which will form caustics)
	amrex::Real a_final = 1.25 * a_collapse; // default is past collapse to see caustics
	pp_cosmo.query("a_final", a_final);

	const amrex::Real t_init = (2.0 / 3.0) * (1.0 / H0) * std::pow(a_init, 1.5);
	const amrex::Real t_final = (2.0 / 3.0) * (1.0 / H0) * std::pow(a_final, 1.5);
	sim.stopTime_ = t_final - t_init;

	const amrex::Geometry &geom = sim.geom[0];
	auto domain = geom.Domain();

	amrex::Print() << "Test resolution (n_cells): "
               	   << domain.length(0) << " x " 
                   << domain.length(1) << " x " 
                   << domain.length(2) << "\n";
	amrex::Print() << "Evolving from a = " << a_init << " to a = " << a_final << " (dt = " << sim.stopTime_ << " s)\n";

	sim.setInitialConditions();
	sim.evolve();

	if (!regression_checked) {
		amrex::Print() << "ERROR: Simulation ended before reaching a = 0.8 * a_collapse!\n";
		status = 1;
	} else {
		status = regression_status;
	}

	return status;
}

// https://bvillasen.github.io/blog/astro/cholla/2019/05/07/zeldovich_pancake.html
// https://github.com/enzo-project/enzo-dev/blob/main/src/enzo/Grid_ZeldovichPancakeInitializeGrid.C

