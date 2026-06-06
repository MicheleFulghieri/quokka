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

#include <AMReX_Math.H>   // for pi
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

    const amrex::Box &indexRange = grid_elem.indexRange_;      // set of the indices of the grid patch 
    const amrex::Array4<double> &state_cc = grid_elem.array_;  // Array4 is a pointer to the data

    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const amrex::Real x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];  // x cell center
		const amrex::Real delta = amplitude * std::cos(k_wave * x);
		const amrex::Real rho = rho_gas / (1.0 - delta);
		const amrex::Real v = - a_init * H_init * (amplitude / k_wave) * std::sin(k_wave * x);
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


template <> void QuokkaSimulation<DMZeldovich>::createInitialCICParticles()  {
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
	
	const int max_lev = 0;
	auto &pc = *CICParticles;                       // take by ref the content of the unique_ptr CICParticles 
	const amrex::Geometry &geom = this->geom[max_lev];
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();      // (dx[0], dx[1], dx[2])
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = geom.ProbHiArray();
	const amrex::Real Lx = prob_hi[0] - prob_lo[0];
 	const amrex::Real Ly = prob_hi[1] - prob_lo[1];
 	const amrex::Real Lz = prob_hi[2] - prob_lo[2];

	const int nparx = 32;
	const int npary = nparx;
	const int nparz = nparx;
	const amrex::Long npartot = static_cast<amrex::Long>(nparx) * npary * nparz;
	
	// Particle masses and distances
	const amrex::Real tot_rho_dm  = Physics_Traits<DMZeldovich>::omega_dm * rho_mean;  // according to the budget
	const amrex::Real tot_mass_dm = tot_rho_dm * Lx * Ly * Lz;
	const amrex::Real part_mass   = tot_mass_dm / npartot;
	const amrex::Real dpartx      = Lx / nparx;
	const amrex::Real dparty      = Ly / npary;
	const amrex::Real dpartz      = Lz / nparz;

	const amrex::Real amplitude  = a_init / a_collapse;   // force collapse at a_collapse
	const amrex::Real k_wave      = (2.0 * amrex::Math::pi<amrex::Real>()) / Lx;

	amrex::Print() << "DM particles initialization along x,y,z: " 
	               << nparx << "x" << npary << "x" << nparz << ")..." << std::endl;

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {  // loop over the tile managed by the current processor	
		const amrex::Box &tile_box = mfi.tilebox();  // integer indexes of the current region
		auto &particles = pc.GetParticles(lev)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];  // particle map
        		
		

	} // end MFITer 


} // end createInitialCICParticles

// aggiungere print delle masse






////////////////////////////////



// template <> void QuokkaSimulation<DMZeldovich>::createInitialCICParticles() {
// 	const int lev = 0;
// 	auto &pc = *CICParticles;   // take by ref the content of the unique_ptr CICParticles 
	
// 	// Geometrical setup
// 	const amrex::Geometry &geom = this->geom[lev];
// 	const auto dx = geom.CellSizeArray();  // (dx[0], dx[1], dx[2])
// 	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
// 	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = geom.ProbHiArray();
// 	const amrex::Real L_x = prob_hi[0] - prob_lo[0];
// 	const amrex::Real L_y = prob_hi[1] - prob_lo[1];
// 	const amrex::Real L_z = prob_hi[2] - prob_lo[2];

// 	// Physical parameters
// 	amrex::Real a_init = PhysicsTraits<DMZeldovich>::a_init;
//     amrex::Real a_collapse = 0.5;
//     amrex::ParmParse pp_cosmo("cosmology");
//     pp_cosmo.query("a_init", a_init);
//     pp_cosmo.query("a_collapse", a_collapse);

// 	const amrex::Real G = PhysicsTraits<DMZeldovich>::gravitational_constant;
// 	const amrex::Real h = PhysicsTraits<DMZeldovich>::hubble_constant;
// 	const amrex::Real Mpc_to_cm = 3.08567758e24;
// 	const amrex::Real H0 = (h * 100.0 * 1e5) / Mpc_to_cm;    // Hubble parameter today (s^-1)
// 	const amrex::Real H_init = H0 * std::pow(a_init, -1.5);  // initial Hubble paramter for EdS from H0
// 	const amrex::Real rho_crit_0 = 3.0 * H0 * H0 / (8.0 * amrex::Math::pi<amrex::Real>() * G);
// 	const amrex::Real rho_mean = rho_crit_0;                 // comoving mean density = critical density (EdS flat universe)

// 	amrex::Box const& domain = geom.Domain();  // return the domain indices 
// 	const auto dom_len = domain.length();      // number of cell along x, y, z (nx, ny, nz)

// 	// Set the number of particles == number of cells (1 particle per cell)
// 	const int n_part_x = dom_len[0];
// 	const int n_part_y = dom_len[1];
// 	const int n_part_z = dom_len[2];
// 	const amrex::Long n_part_total = static_cast<amrex::Long>(n_part_x) * n_part_y * n_part_z;  
// 	const amrex::Real total_mass = rho_mean * L_x * L_y * L_z;  
// 	const amrex::Real particle_mass = total_mass / n_part_total;   
	
// 	const amrex::Real amplitude  = a_init / a_collapse;      // collapse at a_collapse
//  const amrex::Real k_wave = (2.0 * amrex::Math::pi<amrex::Real>()) / L_x;

// 	amrex::Print() << "DM particles initialization...\n";
// 	amrex::Print() << "Mean density set: " << rho_mean << " g/cm^3\n";
// 	amrex::Print() << "Total DM particles created in the domain: " << n_part_total << "\n";
// 	amrex::Print() << "Mass per DM particle: " << particle_mass << " g\n";


// 	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {		
// 		const amrex::Box &tile_box = mfi.tilebox();  // integer indexes of the current region
// 		auto &particles = pc.GetParticles(lev)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];  // particle map
        
// 		amrex::Loop(tile_box, [=, &particles](int i, int j, int k) noexcept {
// 			amrex::Real qx = prob_lo[0] + (i + 0.5) * dx[0];
// 			amrex::Real qy = prob_lo[1] + (j + 0.5) * dx[1];
// 			amrex::Real qz = prob_lo[2] + (k + 0.5) * dx[2];

// 			const amrex::Real displacement = (amplitude / k_wave) * std::sin(k_wave * qx); 
// 			const amrex::Real v_pec        = a_init * H_init * displacement; 

// 			using ParticleType = quokka::CICParticleContainer::ParticleType;  
// 			ParticleType p;  
// 			p.id()  = ParticleType::NextID();
// 			p.cpu() = amrex::ParallelDescriptor::MyProc();
// 			p.pos(0) = qx + displacement;
// 			p.pos(1) = qy;
// 			p.pos(2) = qz;			
// 			p.rdata(quokka::CICParticleMassIdx) = particle_mass;   
// 			p.rdata(quokka::CICParticleVxIdx)   = v_pec;		     
// 			p.rdata(quokka::CICParticleVyIdx)   = 0.0;		     
// 			p.rdata(quokka::CICParticleVzIdx)   = 0.0;		

// 			particles.push_back(p);     // struct of array (SoA)
// 		}); // end amrex::Loop
// 	} // end for amrex::MFIter mfi 
// 	pc.Redistribute();

// 	amrex::Long total_particles = pc.TotalNumberOfParticles();  
// 	amrex::Print() << "Initialization of " << total_particles << " DM particles completed.\n" << std::endl;
	
// }


// template <> void QuokkaSimulation<DMZeldovich>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const
// {
// 	// compute derived variables and save in 'mf'
// 	if (dname == "gpot") {
// 		const int ncomp = ncomp_cc_in;
// 		auto const &phi_arr = phi[lev].const_arrays();
// 		auto output = mf.arrays();
// 		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept { output[bx](i, j, k, ncomp) = phi_arr[bx](i, j, k); });
// 	}
// }


auto problem_main() -> int {
	int status = 0;
	QuokkaSimulation<DMZeldovich> sim;

	const amrex::Geometry &geom = sim.geom[0];
	auto domain = geom.Domain();

	amrex::Print() << "Test resolution (n_cells): "
               	   << domain.length(0) << " x " 
                   << domain.length(1) << " x " 
                   << domain.length(2) << "\n";

	
	sim.setInitialConditions();
	sim.evolve();

    return status;
}



// https://bvillasen.github.io/blog/astro/cholla/2019/05/07/zeldovich_pancake.html
// https://github.com/enzo-project/enzo-dev/blob/main/src/enzo/Grid_ZeldovichPancakeInitializeGrid.C

// invece di calcolarsi la rho_crit, aggiugersela ai Traits basandosi sull H(a) che descrive 
// lo specifico universo