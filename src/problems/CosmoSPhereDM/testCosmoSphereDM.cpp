//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testCosmologicalDarkMatter.cpp
/// \brief Defines a cosmology test problem for CIC particle drifting together with
/// a gas uniform sphere on which is centered


#include "QuokkaSimulation.hpp"
#include "particles/particle_types.hpp"
#include "cosmology/Cosmology.hpp"

#include "AMReX_BLassert.H"
#include <AMReX_ParticleMesh.H>

#include <cmath>



// Struct tag for the templates, with the defalt hydro values
struct CosmoSphereDM {
};

template <> struct quokka::EOS_Traits<CosmoSphereDM> {
    static constexpr Real gamma = 5.0 / 3.0;
    static constexpr Real mean_molecular_weight = C::m_u;
};
 
template <> struct Particle_Traits<CosmoSphereDM> { // CIC from particle_types.hpp
    static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC;
};

template <> struct Physics_Traits<CosmoSphereDM> {
    static constexpr bool is_hydro_enabled        = true;
	static constexpr bool is_cosmology_enabled    = true;
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_radiation_enabled    = false;
	static constexpr bool is_mhd_enabled          = false;
	static constexpr int numMassScalars           = 0;
	static constexpr int numPassiveScalars        = 0;
	static constexpr bool is_dust_enabled         = false;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;

// Cosmology parameters (LCDM)
	static constexpr double omega_m = 0.315;            
	static constexpr double omega_r = 9.2618e-5;
	static constexpr double omega_lambda = 0.685;
	static constexpr double hubble_constant = 0.7;	   // h = 0.7 (H0 = 70 km/s/Mpc)
	static constexpr double a_init = 0.01;		       // start at z = 99
	static constexpr double cosmology_dt_limit = 0.01; // according to the default
};


// Uniform sphere of gas at the domain center
template <> void QuokkaSimulation<CosmoSphereDM>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {

	// set of (x,y,z) indexes and Array4 pointer to the data
	const amrex::Box &indexRange = grid_elem.indexRange_;      
    const amrex::Array4<double> &state_cc = grid_elem.array_;  

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;       // cell dimensions (dx, dy, dz)
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo  = grid_elem.prob_lo_;  // low left physical coordinates
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi  = grid_elem.prob_hi_;  // top right physicsl coordinates

	// Center of the domain
	amrex::Real const X0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
	amrex::Real const Y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
	amrex::Real const Z0 = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {

		// Cell distance from the center
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1];
		amrex::Real const z = prob_lo[2] + (k + static_cast<amrex::Real>(0.5)) * dx[2];
		amrex::Real const r = std::sqrt(std::pow(x - X0, 2) + std::pow(y - Y0, 2) + std::pow(z - Z0, 2));

		amrex::Real const rho_min  = 1.0e-28;
		amrex::Real const rho_max  = 1.0e-24;
		amrex::Real const R_sphere = 3.086e24;   // 1 Mpc
		amrex::Real const R_smooth = 1.543e23;   // 1/20 R_sphere
		amrex::Real const rho = std::max(rho_min, rho_max * ((std::tanh((R_sphere - r) / R_smooth) + 1.0) / 2.0));
		amrex::Real const P  = 1.0e-15;
		amrex::Real const vx = 1.0e6;            // 10 km/s
		amrex::Real const vy = 0.0;
		amrex::Real const vz = 0.0;

		AMREX_ASSERT(!std::isnan(rho));
		AMREX_ASSERT(!std::isnan(P));
		AMREX_ASSERT(!std::isnan(vx));

		state_cc(i, j, k, HydroSystem<CosmoSphereDM>::density_index)        = rho;
		state_cc(i, j, k, HydroSystem<CosmoSphereDM>::x1Momentum_index)     = rho * vx;
		state_cc(i, j, k, HydroSystem<CosmoSphereDM>::x2Momentum_index)     = 0;
		state_cc(i, j, k, HydroSystem<CosmoSphereDM>::x3Momentum_index)     = 0;
		state_cc(i, j, k, HydroSystem<CosmoSphereDM>::internalEnergy_index) = quokka::EOS<CosmoSphereDM>::ComputeEintFromPres(rho, P);
		state_cc(i, j, k, HydroSystem<CosmoSphereDM>::energy_index)         = quokka::EOS<CosmoSphereDM>::ComputeEintFromPres(rho, P) + 0.5 * rho * (vx * vx + vy * vy + vz * vz); // eint + ekin
	});
}

#if AMREX_SPACEDIM == 3
template <> void QuokkaSimulation<CosmoSphereDM>::createInitialCICParticles() {

	const int lev = 0;                               
	const amrex::Geometry &geom = this->geom[lev];	  // retrive geometry

	// GpuArray with the borders of the domain
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = geom.ProbHiArray();

	// Center of the domain
	amrex::Real const X0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
	amrex::Real const Y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
	amrex::Real const Z0 = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);
	amrex::RealVect const center_coords{X0, Y0, Z0};
	amrex::IntVect const center_index = geom.CellIndex(center_coords.dataPtr());

	amrex::Real const R_sphere = 3.086e24; // 1 Mpc
    amrex::Real const rho_max  = 1.0e-24;
    amrex::Real const mass_gas = (4.0 / 3.0) * M_PI * std::pow(R_sphere, 3) * rho_max;

	amrex::Real const mass_dm  = 5.0 * mass_gas; // DM fivefolds the gas
    amrex::Real const vx_dm   = 1.0e6;    
    amrex::Real const vy_dm   = 0.0;
    amrex::Real const vz_dm   = 0.0;

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {  // over the local boxes assigned to this processor
        const amrex::Box& valid_box = mfi.validbox();

		if (valid_box.contains(center_index)) { // only the core of the central cell
			
			// Access the vector of particles of the correct tile
			auto& particles = CICParticles->GetParticles(lev)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
			
			using ParticleType = quokka::CICParticleContainer::ParticleType;  // create the alias
			ParticleType p;
			p.id() = ParticleType::NextID();                   // ID assignement
			p.cpu() = amrex::ParallelDescriptor::MyProc();     // processor assignement

			p.pos(0) = X0;
        	p.pos(1) = Y0;
        	p.pos(2) = Z0;
			p.rdata(quokka::CICParticleMassIdx) = mass_dm;
			p.rdata(quokka::CICParticleVxIdx)   = vx_dm;
			p.rdata(quokka::CICParticleVyIdx)   = vy_dm;
			p.rdata(quokka::CICParticleVzIdx)   = vz_dm;

			particles.push_back(p);  // push the local particle in the local core memory
		}
	}
	CICParticles->Redistribute(); // assign the particle to the right mpi core
}
#endif   // AMREX_SPACEDIM == 3


template <> void QuokkaSimulation<CosmoSphereDM>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const
{
	// compute derived variables and save in 'mf'
	if (dname == "gpot") {
		const int ncomp = ncomp_cc_in;
		auto const &phi_arr = phi[lev].const_arrays();
		auto output = mf.arrays();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept { output[bx](i, j, k, ncomp) = phi_arr[bx](i, j, k); });
	}
}


auto problem_main() -> int {
	int status = 0;

	QuokkaSimulation<CosmoSphereDM> sim;
	sim.setInitialConditions();
	sim.evolve();

    return status;
}




// vedere se spegnere la self gravity

// aggiungere il test





	// if (amrex::ParallelDescriptor::IOProcessor()) { // only processor 0
	// 	using ParticleType = quokka::CICParticleContainer::ParticleType;  // create the alias
	// 	ParticleType p;
	// 	p.id() = ParticleType::NextID();                   // ID assignement
	// 	p.cpu() = amrex::ParallelDescriptor::MyProc();     // processor assignement

	// 	p.pos(0) = X0;
    //     p.pos(1) = Y0;
    //     p.pos(2) = Z0;
	// 	p.rdata(quokka::CICParticleMassIdx) = mass_dm;
	// 	p.rdata(quokka::CICParticleVxIdx)   = vx_dm;
	// 	p.rdata(quokka::CICParticleVyIdx)   = vy_dm;
	// 	p.rdata(quokka::CICParticleVzIdx)   = vz_dm;

	// 	// Place the particle temporary in grid 0 and tile 0 of the processor 0 regardless of the real coordinates
	// 	CICParticles->GetParticles(lev)[std::make_pair(0, 0)].push_back(p);