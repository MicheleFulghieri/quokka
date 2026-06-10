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
	static constexpr amrex::Real drift_vel = 1.0e8;     // 1000 km/s both for gas and DM
};

template <> struct quokka::EOS_Traits<CosmoSphereDM> {
    static constexpr amrex::Real gamma = 5.0 / 3.0;
    static constexpr amrex::Real mean_molecular_weight = C::m_u;
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
	static constexpr amrex::Real omega_m = 0.30966;
	static constexpr amrex::Real omega_r = 9.13896e-05;
	static constexpr amrex::Real omega_lambda = 0.68885;
	static constexpr amrex::Real omega_b = 0.04897;        
	static constexpr amrex::Real omega_dm = 0.26069;
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

		amrex::Real const rho_min  = 1.0e-27;
		amrex::Real const rho_max  = 1.0e-24;
		amrex::Real const R_sphere = 1.543e23;   // 50 kpc
		amrex::Real const R_smooth = 7.715e21;   // 1/20 R_sphere
		amrex::Real const rho = std::max(rho_min, rho_max * ((std::tanh((R_sphere - r) / R_smooth) + 1.0) / 2.0));
		amrex::Real const P = 1.0e-14;
		amrex::Real const T = P * quokka::EOS_Traits<CosmoSphereDM>::mean_molecular_weight / (rho * C::k_B);
		amrex::Real const vx = CosmoSphereDM::drift_vel;
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

	amrex::Real const R_sphere = 3.086e23; // 100 kpc
    amrex::Real const rho_max  = 1.0e-24;
    amrex::Real const mass_gas = (4.0 / 3.0) * M_PI * std::pow(R_sphere, 3) * rho_max;

	amrex::Real const mass_dm  = 5.0 * mass_gas; // DM fivefolds the gas
    amrex::Real const vx_dm   = CosmoSphereDM::drift_vel;    
    amrex::Real const vy_dm   = 0.0;
    amrex::Real const vz_dm   = 0.0;

	if (amrex::ParallelDescriptor::IOProcessor()) {   // only Rank MPI 0: only 1 particle
		using ParticleType = quokka::CICParticleContainer::ParticleType; // alias
		ParticleType p;  
		p.id()  = ParticleType::NextID();               // ID assignement 
		p.cpu() = amrex::ParallelDescriptor::MyProc();  // processor assignement (0 by construction)

		p.pos(0) = X0;
        p.pos(1) = Y0;
        p.pos(2) = Z0;
		p.rdata(quokka::CICParticleMassIdx) = mass_dm;
		p.rdata(quokka::CICParticleVxIdx)   = vx_dm;
		p.rdata(quokka::CICParticleVyIdx)   = vy_dm;
		p.rdata(quokka::CICParticleVzIdx)   = vz_dm;

		amrex::MFIter mfi(state_new_cc_[lev]); // creation of the iterator over the hydro local boxes of processor 0
		if (mfi.isValid()) {   // if processor 0 has at least one level 0 grid
			auto const key = std::make_pair(mfi.index(), mfi.LocalTileIndex());  // extact the local indices of that grid
        	CICParticles->GetParticles(lev)[key].push_back(p);  //  temporarily store the particle in the memory of that box
		} else {
			amrex::Abort("Error: 'IOProcessor has no local grid assigned to level 0!");
		}
	}
	CICParticles->Redistribute(); // assign the particle to the right mpi core according to the physical position
}
#endif   // AMREX_SPACEDIM == 3


auto problem_main() -> int {
	int status = 0;

	QuokkaSimulation<CosmoSphereDM> sim;
	sim.setInitialConditions();
	sim.evolve();

	amrex::Print() << "Computing particle-gas distance for the test...\n";

	// Sphere center as the gas density weighted mean cell center position 
	const int finest_level = sim.finestLevel();
	const amrex::Geometry &geom = sim.geom[finest_level];
	const auto &dx            = geom.CellSizeArray();
	const auto &prob_lo       = geom.ProbLoArray();
	const amrex::MultiFab &mf = sim.state_new_cc_[finest_level];
	
	// Extract Domain information (Physical and Index space)
	const amrex::Box &domain_box = geom.Domain();
	const auto domain_lo = domain_box.smallEnd();
	const auto domain_hi = domain_box.bigEnd();
	const auto prob_hi   = geom.ProbHiArray();

	// Calculate total cells per dimension
	int n_cells_x = domain_hi[0] - domain_lo[0] + 1;
	int n_cells_y = domain_hi[1] - domain_lo[1] + 1;
	int n_cells_z = domain_hi[2] - domain_lo[2] + 1;

	amrex::Real total_mass_density = 0.0;
	amrex::Real sum_x = 0.0;
	amrex::Real sum_y = 0.0;
	amrex::Real sum_z = 0.0;

	// AMReX reductor operator for the 4 sums
	amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_ops;
	amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real> reduce_data(reduce_ops);   // memory preallocation for the sums

	for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox(); // no ghost cells
		auto const &state_arr = mf.array(mfi);  // cell data array

		reduce_ops.eval(box, reduce_data,
			[=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real> {
				amrex::Real cell_center_x = prob_lo[0] + (i + 0.5) * dx[0];  // x center of the cell
				amrex::Real cell_center_y = prob_lo[1] + (j + 0.5) * dx[1];
				amrex::Real cell_center_z = prob_lo[2] + (k + 0.5) * dx[2];
				amrex::Real rho = state_arr(i, j, k, 0);                     // density (0 component of state_cc)

				// Values to be summed (rho*x, rho*y, rho*z, rho)
				return {rho * cell_center_x, rho * cell_center_y, rho * cell_center_z, rho};
			});  // at each call at each cell, partially sums the each cell result
	}  // end mfi interation

	// Structure binding with the partial MPI sums
	auto [g_sum_x, g_sum_y, g_sum_z, g_total_rho] = reduce_data.value();

	// Global MPI reduction
	amrex::ParallelDescriptor::ReduceRealSum(g_sum_x);
	amrex::ParallelDescriptor::ReduceRealSum(g_sum_y);
	amrex::ParallelDescriptor::ReduceRealSum(g_sum_z);
	amrex::ParallelDescriptor::ReduceRealSum(g_total_rho);

	// Gas center
	amrex::Real gas_center_x = g_sum_x / g_total_rho;
	amrex::Real gas_center_y = g_sum_y / g_total_rho;
	amrex::Real gas_center_z = g_sum_z / g_total_rho;

	// Get particle data using the particle descriptor
	const auto [real_data, int_data] = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::CIC)->getParticleDataAtLevel(finest_level);

	const amrex::Real cm_to_Mpc = 1 / (C::parsec * 1.0e6);
	const amrex::Real cm_to_kpc = 1 / (C::parsec * 1.0e3);

	if (amrex::ParallelDescriptor::IOProcessor()) {

		amrex::Print() << "\n=================== DOMAIN & RESOLUTION ===================\n"
					   << "  - Cell Resolution (dx, dy, dz) : (" << dx[0] * cm_to_kpc << ", " << dx[1] * cm_to_kpc << ", " << dx[2] * cm_to_kpc << ") kpc\n"
					   << "  - Domain Cells (Nx, Ny, Nz)    : (" << n_cells_x << ", " << n_cells_y << ", " << n_cells_z << ") cells\n"
					   << "  - Domain Size X [Lo / Hi]      : [" << prob_lo[0] * cm_to_Mpc << " / " << prob_hi[0] * cm_to_Mpc << "] Mpc\n"
					   << "  - Domain Size Y [Lo / Hi]      : [" << prob_lo[1] * cm_to_Mpc << " / " << prob_hi[1] * cm_to_Mpc << "] Mpc\n"
					   << "  - Domain Size Z [Lo / Hi]      : [" << prob_lo[2] * cm_to_Mpc << " / " << prob_hi[2] * cm_to_Mpc << "] Mpc\n"
					   << "-----------------------------------------------------------\n";

		if (real_data.size() > 0) {   // there is the particle
			const auto p = real_data[0];
			const amrex::Real px = p[0]; // position x
			const amrex::Real py = p[1]; // position y
			const amrex::Real pz = p[2]; // position z

			// Euclidean distance
			amrex::Real shift_x = px - gas_center_x;
			amrex::Real shift_y = py - gas_center_y;
			amrex::Real shift_z = pz - gas_center_z;
			amrex::Real shift_mpc = std::sqrt(shift_x * shift_x + shift_y * shift_y + shift_z* shift_z);
			amrex::Real shift_cell = shift_mpc / dx[0];
			amrex::Real tolerance_cell = 2;  
			amrex::Real tolerance_mpc = tolerance_cell * dx[0];	
			
			amrex::Print() << "  - Gas Center of Mass           : (" << gas_center_x * cm_to_Mpc << ", " << gas_center_y * cm_to_Mpc << ", " << gas_center_z * cm_to_Mpc << ") Mpc\n"
						   << "  - Particle Position            : (" << px * cm_to_Mpc << ", " << py * cm_to_Mpc << ", " << pz * cm_to_Mpc << ") Mpc\n"
						   << "  - Calculated Distance          : " << shift_mpc * cm_to_kpc << " kpc (" << shift_cell << " cells)\n"
						   << "  - Allowed Tolerance            : " << tolerance_mpc * cm_to_kpc << " kpc (" << tolerance_cell << " cells)\n"
						   << "-----------------------------------------------------------\n";

			if (shift_cell > tolerance_cell) {
			amrex::Print() << "\n========================================================\n"
							   << "[TEST FAILED]: CosmoSphereDM misalignment detected!\n"
							   << "  - Particle position : (" << px * cm_to_Mpc << ", " << py * cm_to_Mpc << ", " << pz * cm_to_Mpc << ") Mpc\n"
							   << "  - Gas center of mass: (" << gas_center_x * cm_to_Mpc << ", " << gas_center_y * cm_to_Mpc << ", " << gas_center_z * cm_to_Mpc << ") Mpc\n"
							   << "  - Absolute distance : " << shift_mpc * cm_to_kpc << " kpc (Tol: " << tolerance_mpc * cm_to_kpc << " kpc)\n"
							   << "  - Relative distance : " << shift_cell << " cell widths (Tol: " << tolerance_cell << " cell widths)\n"
							   << "========================================================\n\n";	
			status = 1;
			} // end if shift > tolerance
			else {
				amrex::Print() << "\n========================================================\n"
							   << "[TEST PASSED]: CosmoSphereDM alignment check successful!\n"
							   << "  - Particle position : (" << px * cm_to_Mpc << ", " << py * cm_to_Mpc << ", " << pz * cm_to_Mpc << ") Mpc\n"
							   << "  - Gas center of mass: (" << gas_center_x * cm_to_Mpc << ", " << gas_center_y * cm_to_Mpc << ", " << gas_center_z * cm_to_Mpc << ") Mpc\n"
							   << "  - Absolute distance : " << shift_mpc * cm_to_kpc << " kpc (Tol: " << tolerance_mpc * cm_to_kpc << " kpc)\n"
							   << "  - Relative distance : " << shift_cell << " cell widths (Tol: " << tolerance_cell << " cell widths)\n"
							   << "========================================================\n\n";
				status = 0;
			} // end else (shift < tolerance)
		} // end real_data.size() > 0)
		else {
			amrex::Print() << "[TEST FAILED]: Particle not found. ";
			status = 1;
			}
		amrex::Print() << "  Gas sphere and DM drift velocity = " << CosmoSphereDM::drift_vel / 100000 << " km/s \n";
	} // end if IOProcessor
	
	// MPI Broadcast, update status from IOprocessor to the others
	amrex::ParallelDescriptor::Bcast(&status, 1, amrex::ParallelDescriptor::IOProcessorNumber());
    return status;
}


