//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testCosmologicalDarkMatter.cpp
/// \brief DM-only Zel'dovich test (Zel’dovich, 1970).
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
	static constexpr amrex::Real omega_b = 0.001;            
	static constexpr amrex::Real omega_dm = 0.999;			// DM-only test
	static constexpr amrex::Real hubble_constant = 0.7;	    // h = 0.7 (H0 = 70 km/s/Mpc)
	static constexpr amrex::Real a_init = 0.01;		        // start at z = 99
	static constexpr amrex::Real cosmology_dt_limit = 0.01; // according to the default
};


template <> void QuokkaSimulation<DMZeldovich>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<amrex::Real> &state_cc = grid_elem.array_;

	amrex::Real a_init = PhysicsTraits<DMZeldovich>::a_init;
	amrex::ParmParse pp_cosmo("cosmology");
	pp_cosmo.query("a_init", a_init);

	// Tune mean density: the simulation comoving mean density sets the effective
	// critical density for this EdS test and the Hubble parameter is scaled accordingly,
	// to keep consistency in the Friedman solver.
	const amrex::Real rho_crit_sim = this->comoving_mean_density_;
	const amrex::Real G = PhysicsTraits<DMZeldovich>::gravitational_constant;
	const amrex::Real h_real  = PhysicsTraits<DMZeldovich>::hubble_constant;
	const amrex::Real pc_to_cm = C::parsec;
	const amrex::Real kpc_to_cm = pc_to_cm * 1.0e3;
	const amrex::Real Mpc_to_cm = pc_to_cm * 1.0e6;
	const amrex::Real H0_real = (h_real * 100.0 * 1e5) / Mpc_to_cm;    // Hubble parameter today (s^-1)
	const amrex::Real rho_crit_0_real = 3.0 * H0_real * H0_real / (8.0 * amrex::Math::pi<amrex::Real>() * G);
	const amrex::Real mass_reduction_fraction = rho_crit_sim / rho_crit_0_real;

	const amrex::Real h_sim  = h_real * std::sqrt(mass_reduction_fraction);
	const amrex::Real H0_sim = (h_sim * 100.0 * 1e5) / Mpc_to_cm;
	const amrex::Real H_init = H0_sim * std::pow(a_init, -1.5);  // initial Hubble paramter for EdS from H0
	const amrex::Real rho_sim = rho_crit_sim;
	const amrex::Real omega_b = Physics_Traits<DMZeldovich>::omega_b;
	const amrex::Real rho_b   = omega_b * rho_sim;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<DMZeldovich>::density_index) = rho_b;
		state_cc(i, j, k, HydroSystem<DMZeldovich>::x1Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<DMZeldovich>::x2Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<DMZeldovich>::x3Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<DMZeldovich>::energy_index) = 0;
		state_cc(i, j, k, HydroSystem<DMZeldovich>::internalEnergy_index) = 0;
	});
	amrex::Print() << "\nInitialized a floor hydro background for the timestep." << std::endl;
	amrex::Print() << "Omega_gas                      : " <<  omega_b << std::endl;
	amrex::Print() << "Gas density                    : " <<  rho_b << "g/cm^3" << std::endl;
	amrex::Print() << "Total matter density (gas + DM): " <<  rho_sim << "g/cm^3" << std::endl;
}

template <> void QuokkaSimulation<DMZeldovich>::createInitialCICParticles() {
	amrex::Real a_init = PhysicsTraits<DMZeldovich>::a_init;
	amrex::Real a_collapse = 0.5;
	amrex::ParmParse pp_cosmo("cosmology");
	pp_cosmo.query("a_init", a_init);
	pp_cosmo.query("a_collapse", a_collapse);

	// Consistent EdS with tuned critical density
	const amrex::Real rho_crit_sim = this->comoving_mean_density_;
	const amrex::Real G = PhysicsTraits<DMZeldovich>::gravitational_constant;
	const amrex::Real h_real  = PhysicsTraits<DMZeldovich>::hubble_constant;
	const amrex::Real pc_to_cm = C::parsec;
	const amrex::Real kpc_to_cm = pc_to_cm * 1.0e3;
	const amrex::Real Mpc_to_cm = pc_to_cm * 1.0e6;
	const amrex::Real H0_real = (h_real * 100.0 * 1e5) / Mpc_to_cm;    // Hubble parameter today (s^-1)
	const amrex::Real rho_crit_0_real = 3.0 * H0_real * H0_real / (8.0 * amrex::Math::pi<amrex::Real>() * G);
	const amrex::Real mass_reduction_fraction = rho_crit_sim / rho_crit_0_real;

	const amrex::Real h_sim  = h_real * std::sqrt(mass_reduction_fraction);
	const amrex::Real H0_sim = (h_sim * 100.0 * 1e5) / Mpc_to_cm;
	const amrex::Real H_init = H0_sim * std::pow(a_init, -1.5);  // initial Hubble paramter for EdS from H0
	const amrex::Real rho_sim = rho_crit_sim;

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
	const amrex::Real tot_rho_dm  = Physics_Traits<DMZeldovich>::omega_dm * rho_sim;  // according to the budget
	const amrex::Real tot_mass_dm = tot_rho_dm * Lx * Ly * Lz;
	const amrex::Real part_mass   = tot_mass_dm / npartot;
	const amrex::Real dpartx      = Lx / nparx;
	const amrex::Real dparty      = Ly / npary;
	const amrex::Real dpartz      = Lz / nparz;

	const amrex::Real amplitude   = a_init / a_collapse;   // force collapse at a_collapse
	const amrex::Real k_wave      = (2.0 * amrex::Math::pi<amrex::Real>()) / Lx;

	amrex::Print() << "\n=== DM Zel'dovich initialization summary ===" << std::endl;
	amrex::Print() << "Domain size (x, y, z): "
		<< (Lx / kpc_to_cm) << " kpc, "
		<< (Ly / kpc_to_cm) << " kpc, "
		<< (Lz / kpc_to_cm) << " kpc" << std::endl;
	amrex::Print() << "Particle grid: " << nparx << " x " << npary << " x " << nparz << " = " << npartot << " particles" << std::endl;
	amrex::Print() << "Simulation mean density: " << rho_sim << " g/cm^3" << std::endl;
	amrex::Print() << "Equivalent real EdS critical density: " << rho_crit_0_real << " g/cm^3" << std::endl;
	amrex::Print() << "Mass reduction fraction: " << mass_reduction_fraction << " (rho_sim / rho_crit_0_real)" << std::endl;
	amrex::Print() << "Hubble parameter scaling: h_sim = sqrt(mass_fraction) * h_real = "
		<< h_sim << " (h_real = " << h_real << ")" << std::endl;
	amrex::Print() << "Initial Hubble rate: H0_sim = " << H0_sim << " s^-1" << std::endl;
	amrex::Print() << "Total DM mass in domain: " << tot_mass_dm << " g" << std::endl;
	amrex::Print() << "Mass per DM particle: " << part_mass << " g" << std::endl;
	amrex::Print() << "Perturbation amplitude: " << (amplitude / k_wave) / kpc_to_cm << " kpc" << std::endl;
	amrex::Print() << "===========================================" << std::endl;

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		amrex::Box const& valid_box = mfi.validbox();   // current valid eulerian box
		auto &particles = pc.GetParticles(lev)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];  // local particle container of the current box
		
		// Calculate the unperturbed Lagrangian positions
		for (int i = 0; i < nparx; ++i) {
			for (int j = 0; j < npary; ++j) {
				for (int k = 0; k < nparz; ++k) {
					amrex::Real qx = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dpartx;
					amrex::Real qy = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dparty;
					amrex::Real qz = prob_lo[2] + (k + static_cast<amrex::Real>(0.5)) * dpartz;

					// Shift qx to centre the collapse (qx_center = qx - L_box/2)
        			const amrex::Real qx_center    = qx - 0.5 * Lx;
        			const amrex::Real displacement = - (amplitude / k_wave) * std::sin(k_wave * qx_center);
        			const amrex::Real v_pec        = a_init * H_init * displacement;

					// Ensure periodic boundary wrapping (if the perturbation dispaces particles out of physical domain)
					amrex::Real x_perturbed = qx + displacement;
					amrex::Real y_perturbed = qy;  
					amrex::Real z_perturbed = qz;

					x_perturbed -= Lx * std::floor((x_perturbed - prob_lo[0]) / Lx);
					y_perturbed -= Ly * std::floor((y_perturbed - prob_lo[1]) / Ly);  // just for security, since only x is perturbed				
					z_perturbed -= Lz * std::floor((z_perturbed - prob_lo[2]) / Lz);

					// Convert the perturbed positions into indices
					amrex::RealVect part_pos{x_perturbed, y_perturbed, z_perturbed};
					amrex::IntVect cell_idxs = geom.CellIndex(part_pos.dataPtr());

					if (valid_box.contains(cell_idxs)) {  // if the particle pos idx belongs to the current process
						using ParticleType = quokka::CICParticleContainer::ParticleType; // alias
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
					} // end if (valid_box.contains(cell_idxs))
				} // end for int k < nparz 
			} // end for int j < npary 
		} // end for int i < nparx 
	} // end MFIter mfi(state_new_cc_[lev])
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

	// Consistent EdS with tuned critical density
	const amrex::Real rho_crit_sim = sim.comoving_mean_density_;
	const amrex::Real G = PhysicsTraits<DMZeldovich>::gravitational_constant;
	const amrex::Real h_real  = PhysicsTraits<DMZeldovich>::hubble_constant;
	const amrex::Real pc_to_cm = C::parsec;
	const amrex::Real kpc_to_cm = pc_to_cm * 1.0e3; 
	const amrex::Real Mpc_to_cm = pc_to_cm * 1.0e6; 
	const amrex::Real H0_real = (h_real * 100.0 * 1e5) / Mpc_to_cm;    // Hubble parameter today (s^-1)
	const amrex::Real rho_crit_0_real = 3.0 * H0_real * H0_real / (8.0 * amrex::Math::pi<amrex::Real>() * G);
	const amrex::Real mass_reduction_fraction = rho_crit_sim / rho_crit_0_real;

	const amrex::Real h_sim  = h_real * std::sqrt(mass_reduction_fraction);
	const amrex::Real H0_sim = (h_sim * 100.0 * 1e5) / Mpc_to_cm;
	const amrex::Real H_init = H0_sim * std::pow(a_init, -1.5);  // initial Hubble paramter for EdS from H0


	// Evolve until a_final (which will form caustics)
	amrex::Real a_final = 1.15 * a_collapse; // default is past collapse to see caustics
	pp_cosmo.query("a_final", a_final);

	const amrex::Real t_init  = (2.0 / 3.0) * (1.0 / H0_sim) * std::pow(a_init, 1.5);
	const amrex::Real t_final = (2.0 / 3.0) * (1.0 / H0_sim) * std::pow(a_final, 1.5);
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

