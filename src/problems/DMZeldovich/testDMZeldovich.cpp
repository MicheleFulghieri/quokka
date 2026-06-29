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
	static constexpr amrex::Real hubble_constant = 0.7;	    // h = 0.7 (H0 = 70 km/s/Mpc)
	static constexpr amrex::Real a_init = 0.01;		        // start at z = 99
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
	               << "(" << nparx << "x" << npary << "x" << nparz << ")..." << std::endl;
	amrex::Print() << "Mass per DM particle: " 
	               << part_mass << "g" << std::endl;

	if (amrex::ParallelDescriptor::IOProcessor()) {
		using ParticleType = quokka::CICParticleContainer::ParticleType;
		
		// Get the first local tile key for processor 0 to temporarily host all the particles
		amrex::MFIter mfi(state_new_cc_[lev]);  // iteration only on the IOproc tiles
		if (mfi.isValid()) {  // place temporarily all the particle in the firts valid tile waiting for Redistribute()
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

						// Ensure periodic boundary wrapping (if the perturbation dispaces particles out of physical domain)
						amrex::Real x_perturbed = qx + displacement;
						while (x_perturbed < prob_lo[0]) x_perturbed += Lx;
						while (x_perturbed >= prob_hi[0]) x_perturbed -= Lx;

						amrex::Real y_perturbed = qy;  // just for security, since only x is perturbed
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


template <> void QuokkaSimulation<DMZeldovich>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const
{
	if (dname == "gpot") {
		const int ncomp = ncomp_cc_in;
		auto const &phi_arr = phi[lev].const_arrays();
		auto output = mf.arrays();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept { output[bx](i, j, k, ncomp) = phi_arr[bx](i, j, k); });
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

	return status;
}

// https://bvillasen.github.io/blog/astro/cholla/2019/05/07/zeldovich_pancake.html
// https://github.com/enzo-project/enzo-dev/blob/main/src/enzo/Grid_ZeldovichPancakeInitializeGrid.C

