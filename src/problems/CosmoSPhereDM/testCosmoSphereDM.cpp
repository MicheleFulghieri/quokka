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

// Cosmology parameters (EdS)
	static constexpr amrex::Real omega_m = 1.0;
	static constexpr amrex::Real omega_r = 0.0;
	static constexpr amrex::Real omega_lambda = 0.0;
	static constexpr amrex::Real omega_b = 0.17;        
	static constexpr amrex::Real omega_dm = 0.83;
	static constexpr amrex::Real hubble_constant = 0.7;	   // h = 0.7 (H0 = 70 km/s/Mpc)
	static constexpr amrex::Real a_init = 0.01;		       // start at z = 99
	static constexpr amrex::Real cosmology_dt_limit = 0.01; // according to the default
};


// Uniform sphere of gas at the domain center
template <> void QuokkaSimulation<CosmoSphereDM>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {

	// set of (x,y,z) indexes and Array4 pointer to the data
	const amrex::Box &indexRange = grid_elem.indexRange_;      
    const amrex::Array4<amrex::Real> &state_cc = grid_elem.array_;  

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;       // cell dimensions (dx, dy, dz)
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo  = grid_elem.prob_lo_;  // low left physical coordinates
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi  = grid_elem.prob_hi_;  // top right physicsl coordinates

	// Center of the domain
	amrex::Real const X0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
	amrex::Real const Y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
	amrex::Real const Z0 = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);

	// Gas physical paramters
	amrex::Real const rho_min  = 1.0e-27;
	amrex::Real const rho_max  = 1.0e-24;
	amrex::Real const P = 1.0e-14;
	amrex::Real const vx = CosmoSphereDM::drift_vel;
	amrex::Real const vy = 0.0;
	amrex::Real const vz = 0.0;
	amrex::Real R_sphere = 3.086e23;       // default: 100 kpc
	amrex::Real R_smooth = 6.172e22;       // default: 1/5 R_sphere (20 kpc)
	amrex::ParmParse pp_sphere("sphere");               
	pp_sphere.query("R_sphere", R_sphere);
	pp_sphere.query("R_smooth", R_smooth);
	// for isothermal -> amrex::Real const T = P * quokka::EOS_Traits<CosmoSphereDM>::mean_molecular_weight / (rho * C::k_B);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {

		// Cell distance from the center
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1];
		amrex::Real const z = prob_lo[2] + (k + static_cast<amrex::Real>(0.5)) * dx[2];
		amrex::Real const r = std::sqrt(std::pow(x - X0, 2) + std::pow(y - Y0, 2) + std::pow(z - Z0, 2));

		amrex::Real const rho = std::max(rho_min, rho_max * ((std::tanh((R_sphere - r) / R_smooth) + 1.0) / 2.0));
		//amrex::Real const rho = (r <= R_sphere) ? rho_max : rho_min;  // abrupt transition

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

	amrex::Real R_sphere = 3.086e23; // 100 kpc
	amrex::ParmParse pp_sphere("sphere");               
	pp_sphere.query("R_sphere", R_sphere);
    amrex::Real const rho_max  = 1.0e-24;
    amrex::Real const mass_gas = (4.0 / 3.0) * M_PI * std::pow(R_sphere, 3) * rho_max;
	amrex::Real const mass_dm  = 5.0 * mass_gas; // DM fivefolds the gas
    amrex::Real const vx_dm   = CosmoSphereDM::drift_vel;    
    amrex::Real const vy_dm   = 0.0;
    amrex::Real const vz_dm   = 0.0;

	// Initialize the particle at domain center
	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) { // iterator instanziation state_new_cc_[lev] contains the hydro MultiFabs (from QuokkaSimulation.hpp and simulation.hpp)
		amrex::Box const& valid_box = mfi.validbox();  // extract the current valid Box
		
		using ParticleType = quokka::CICParticleContainer::ParticleType; // alias
		ParticleType p;  

		if (valid_box.contains(center_index)) {        // select the processor managing the center of the box
			p.id()  = ParticleType::NextID();               // ID assignement 
			p.cpu() = amrex::ParallelDescriptor::MyProc();  // processor assignement (0 by construction)
			p.pos(0) = X0;
        	p.pos(1) = Y0;
        	p.pos(2) = Z0;
			p.rdata(quokka::CICParticleMassIdx) = mass_dm;
			p.rdata(quokka::CICParticleVxIdx)   = vx_dm;
			p.rdata(quokka::CICParticleVyIdx)   = vy_dm;
			p.rdata(quokka::CICParticleVzIdx)   = vz_dm;

			auto const key = std::make_pair(mfi.index(), mfi.LocalTileIndex());
			CICParticles->GetParticles(lev)[key].push_back(p);
		}  // end if valid_box.contains
	}  // end for mfi(state_new_cc_[lev])
	CICParticles->Redistribute(); // assign the particle to the right mpi core according to the physical position
} // end createInitialCICParticles()
#endif   // AMREX_SPACEDIM == 3

template <> void QuokkaSimulation<CosmoSphereDM>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/) {
	// Tag cells for refinement
	const amrex::Real eta_threshold = 0.6; // gradient refinement threshold
	const amrex::Real rho_min = 1.0e-27;   // minimum density for refinement

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_cc_[lev].const_array(mfi);   // array for read hydro
		const auto tag = tags.array(mfi);                         // array for write data

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const int rho_idx = HydroSystem<CosmoSphereDM>::density_index;
			amrex::Real const rho = state(i, j, k, rho_idx);
			amrex::Real const rho_xplus  = state(i + 1, j, k, rho_idx);
			amrex::Real const rho_xminus = state(i - 1, j, k, rho_idx);
			amrex::Real const rho_yplus  = state(i, j + 1, k, rho_idx);
			amrex::Real const rho_yminus = state(i, j - 1, k, rho_idx);
			amrex::Real const rho_zplus  = state(i, j , k + 1, rho_idx);
			amrex::Real const rho_zminus = state(i, j, k -1, rho_idx);

			amrex::Real const del_x = std::max(std::abs(rho_xplus - rho), std::abs(rho - rho_xminus));
			amrex::Real const del_y = std::max(std::abs(rho_yplus - rho), std::abs(rho - rho_yminus));
			amrex::Real const del_z = std::max(std::abs(rho_zplus - rho), std::abs(rho - rho_zminus));

			amrex::Real const gradient_indicator = std::max({del_x, del_y, del_z}) / std::max(rho, rho_min);  // std::max({del_x, del_y, del_z}) via initializer list

			if (rho > 5.0e-27 && gradient_indicator > eta_threshold) { // avoid refine the background via rho lower threshold
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});  // end amrex::ParallelFor over the box

	} // end amrex::MFIter mfi
}

auto problem_main() -> int {
	int status = 0;

	QuokkaSimulation<CosmoSphereDM> sim;
	sim.setInitialConditions();
	sim.evolve();

	amrex::Print() << "Computing particle-gas distance for the test...\n";

	// Sphere center as the gas density weighted mean cell center position 
    // Level 0 to mathematically avoid double-counting in AMR and cover the whole universe
	const int lev = 0;
	const amrex::Geometry &geom = sim.geom[lev];
	const auto dx            = geom.CellSizeArray();
	const auto prob_lo       = geom.ProbLoArray();
	const auto prob_hi       = geom.ProbHiArray();
	const amrex::MultiFab &mf = sim.state_new_cc_[lev];
	amrex::Real Lx = prob_hi[0] - prob_lo[0];
	amrex::Real Ly = prob_hi[1] - prob_lo[1];
	amrex::Real Lz = prob_hi[2] - prob_lo[2];
	
	// Extract Domain information (Physical and Index space)
	const amrex::Box &domain_box = geom.Domain();
	const auto domain_lo = domain_box.smallEnd();
	const auto domain_hi = domain_box.bigEnd();

	// Calculate total cells per dimension on level 0
	int n_cells_x = domain_hi[0] - domain_lo[0] + 1;
	int n_cells_y = domain_hi[1] - domain_lo[1] + 1;
	int n_cells_z = domain_hi[2] - domain_lo[2] + 1;

	// AMReX reductor operator for the 6 sums: (rho*cos(theta_x), rho*sin(theta_x), ...)
	amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_ops;
	amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real> reduce_data(reduce_ops);   // memory preallocation for the sums

	for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox(); // no ghost cells
		auto const &state_arr = mf.array(mfi);  // cell data array

		reduce_ops.eval(box, reduce_data,
			[=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real> {
				amrex::Real cell_center_x = prob_lo[0] + (i + 0.5) * dx[0];  // x center of the cell
				amrex::Real cell_center_y = prob_lo[1] + (j + 0.5) * dx[1];
				amrex::Real cell_center_z = prob_lo[2] + (k + 0.5) * dx[2];
				amrex::Real rho = state_arr(i, j, k, 0);                     // density (0 component of state_cc)

				// Compute the exact center of mass in a periodic domain using the phase of the first Fourier mode
				amrex::Real theta_x = 2.0 * M_PI * (cell_center_x - prob_lo[0]) / Lx;  // map coordinates to [0, 2pi] relative to the left edge of the domain
				amrex::Real theta_y = 2.0 * M_PI * (cell_center_y - prob_lo[1]) / Ly;  
				amrex::Real theta_z = 2.0 * M_PI * (cell_center_z - prob_lo[2]) / Lz;  

				// Values to be summed: first Fourier mode components (rho*cos(theta_x), rho*sin(theta_x), ...)
				return {rho * std::cos(theta_x), rho * std::sin(theta_x),
						rho * std::cos(theta_y), rho * std::sin(theta_y), 
						rho * std::cos(theta_z), rho * std::sin(theta_z)};
			});  // at each call at each cell, partially sums the each cell result
	}  // end mfi interation

	// Structure binding with the partial MPI sums
	auto [cos_x, sin_x, cos_y, sin_y, cos_z, sin_z] = reduce_data.value(); // this includes global MPI_Allreduce across all processors

	// Center of mass in periodic theta coords
	amrex::Real theta_cmx = std::atan2(sin_x, cos_x);  // arctg2(y,x) account for the relative x and y signs
	amrex::Real theta_cmy = std::atan2(sin_y, cos_y);
	amrex::Real theta_cmz = std::atan2(sin_z, cos_z);

	// Back to x, y, z coords
	amrex::Real gas_center_x = prob_lo[0] + std::fmod(theta_cmx / (2.0 * M_PI) * Lx + Lx, Lx);
	amrex::Real gas_center_y = prob_lo[1] + std::fmod(theta_cmy / (2.0 * M_PI) * Ly + Ly, Ly);
	amrex::Real gas_center_z = prob_lo[2] + std::fmod(theta_cmz / (2.0 * M_PI) * Lz + Lz, Lz);

	// Get particle data using the particle descriptor

    // Impossible default values
    amrex::Real px = -1e30, py = -1e30, pz = -1e30;

	for (int lev_p = 0; lev_p <= sim.finestLevel(); ++lev_p) {
		const auto [real_data, int_data] = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::CIC)->getParticleDataAtLevel(lev_p);  // get particle in the particle descriptor
		if (real_data.size() > 0) {   // If the particle is found locally on this rank
			const auto p = real_data[0];
			px = p[0];
            py = p[1]; 
            pz = p[2]; 
			break;  // exit when the particle is found
		}
    }
    
    // Broadcast / reduce particle coordinates so all processors know where it is
    amrex::ParallelDescriptor::ReduceRealMax(px);
    amrex::ParallelDescriptor::ReduceRealMax(py);
    amrex::ParallelDescriptor::ReduceRealMax(pz);

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

		if (px > -1e29) {   // there is the particle
			// Euclidean distance
			amrex::Real shift_x = px - gas_center_x;
			amrex::Real shift_y = py - gas_center_y;
			amrex::Real shift_z = pz - gas_center_z;
			shift_x -= Lx * std::round(shift_x / Lx);  // wrap the shift to [-Lx/2, Lx/2]
			shift_y -= Ly * std::round(shift_y / Ly);
			shift_z -= Lz * std::round(shift_z / Lz);
			amrex::Real shift_mpc = std::sqrt(shift_x * shift_x + shift_y * shift_y + shift_z* shift_z) * cm_to_Mpc;
			amrex::Real shift_cell = shift_mpc / (dx[0] * cm_to_Mpc);
			amrex::Real tolerance_cell = 2;  
			amrex::Real tolerance_mpc = tolerance_cell * dx[0] * cm_to_Mpc;	
			
			amrex::Print() << "  - Gas Center of Mass           : (" << gas_center_x * cm_to_Mpc << ", " << gas_center_y * cm_to_Mpc << ", " << gas_center_z * cm_to_Mpc << ") Mpc\n"
						   << "  - Particle Position            : (" << px * cm_to_Mpc << ", " << py * cm_to_Mpc << ", " << pz * cm_to_Mpc << ") Mpc\n"
						   << "  - Calculated Distance          : " << shift_mpc * 1e3 << " kpc (" << shift_cell << " cells)\n"
						   << "  - Allowed Tolerance            : " << tolerance_mpc * 1e3 << " kpc (" << tolerance_cell << " cells)\n"
						   << "-----------------------------------------------------------\n";

			if (shift_cell > tolerance_cell) {
			amrex::Print() << "\n========================================================\n"
							   << "[TEST FAILED]: CosmoSphereDM misalignment detected!\n"
							   << "  - Particle position : (" << px * cm_to_Mpc << ", " << py * cm_to_Mpc << ", " << pz * cm_to_Mpc << ") Mpc\n"
							   << "  - Gas center of mass: (" << gas_center_x * cm_to_Mpc << ", " << gas_center_y * cm_to_Mpc << ", " << gas_center_z * cm_to_Mpc << ") Mpc\n"
							   << "  - Absolute distance : " << shift_mpc * 1e3 << " kpc (Tol: " << tolerance_mpc * 1e3 << " kpc)\n"
							   << "  - Relative distance : " << shift_cell << " cell widths (Tol: " << tolerance_cell << " cell widths)\n"
							   << "========================================================\n\n";	
			status = 1;
			} // end if shift > tolerance
			else {
				amrex::Print() << "\n========================================================\n"
							   << "[TEST PASSED]: CosmoSphereDM alignment check successful!\n"
							   << "  - Particle position : (" << px * cm_to_Mpc << ", " << py * cm_to_Mpc << ", " << pz * cm_to_Mpc << ") Mpc\n"
							   << "  - Gas center of mass: (" << gas_center_x * cm_to_Mpc << ", " << gas_center_y * cm_to_Mpc << ", " << gas_center_z * cm_to_Mpc << ") Mpc\n"
							   << "  - Absolute distance : " << shift_mpc * 1e3 << " kpc (Tol: " << tolerance_mpc * 1e3 << " kpc)\n"
							   << "  - Relative distance : " << shift_cell << " cell widths (Tol: " << tolerance_cell << " cell widths)\n"
							   << "========================================================\n\n";
				status = 0;
			} // end else (shift < tolerance)
		} // end px > -1e29
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


