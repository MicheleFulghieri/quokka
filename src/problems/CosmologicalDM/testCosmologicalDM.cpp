//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testCosmologicalDarkMatter.cpp
/// \brief Defines a test problem including cosmology and dark matter in 
/// an EdS universe
///

#include "QuokkaSimulation.hpp"
#include "particles/particle_types.hpp"
#include "cosmology/Cosmology.hpp"

#include <AMReX_Math.H>   // for pi
#include <AMReX_ParticleMesh.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <utility>        // for std::make_pairconst 
#include <cmath>          // for std::sin, std::floor


// Global variable for the stutus of the test
static amrex::Real pos_error_at_08     = 0.0;  // NOLINT
static amrex::Real density_error_at_08 = 0.0;  // NOLINT

// Struct tag for the templates, with the defalt hydro values
struct DMExpansionTest{
	static constexpr amrex::Real rho_gas_default = 1.0e-30;    
	static constexpr amrex::Real P_gas_default   = 1.0e-40;    // low pressure (enough to have low sound speed and thus small dt)
};


template <> struct quokka::EOS_Traits<DMExpansionTest> {
	static constexpr amrex::Real gamma = 5.0 / 3.0;
	static constexpr amrex::Real mean_molecular_weight = C::m_u;
};

// Traits: physics and particles
template <> struct Particle_Traits<DMExpansionTest>{  // CIC from particle_types.hpp
    static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC;
};


// Physics traits
template <> struct Physics_Traits<DMExpansionTest> {
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
	static constexpr double omega_m = 1.0;             // EdS universe
	static constexpr double omega_r = 0.0;
	static constexpr double omega_lambda = 0.0;
	static constexpr double hubble_constant = 0.7;	   // h = 0.7 (H0 = 70 km/s/Mpc)
	static constexpr double a_init = 0.01;		       // start at z = 99
	static constexpr double cosmology_dt_limit = 0.01; // according to the default
};


// Grid initialization (only background, hydro only in the floor state_cc)
template <> void QuokkaSimulation<DMExpansionTest>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {

    // Floor hydro values to avoid nan: default 
    amrex::Real rho_floor = DMExpansionTest::rho_gas_default;
    amrex::Real p_floor   = DMExpansionTest::P_gas_default;
    amrex::Real gamma = quokka::EOS_Traits<DMExpansionTest>::gamma;

	// Initial conditions: precedence to user-entered values (override if present in the input file)
	amrex::ParmParse pp("problem");
	pp.query("rho0_gas", rho_floor);
	pp.query("P0_gas", p_floor);
	pp.query("gamma", gamma);

    const amrex::Box &indexRange = grid_elem.indexRange_;       // set of the indices of the grid patch (e.g. from 0 to 31 in x, y, z)
    const amrex::Array4<double> &state_cc = grid_elem.array_;  // Array4 is a pointer to the data

    // Parallel loop over the spatial indices on GPU
    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
        state_cc(i, j, k, HydroSystem<DMExpansionTest>::density_index) = rho_floor; 
        state_cc(i, j, k, HydroSystem<DMExpansionTest>::x1Momentum_index) = 0;
        state_cc(i, j, k, HydroSystem<DMExpansionTest>::x2Momentum_index) = 0;
        state_cc(i, j, k, HydroSystem<DMExpansionTest>::x3Momentum_index) = 0;
        state_cc(i, j, k, HydroSystem<DMExpansionTest>::energy_index) = p_floor / (gamma - 1.0);
        state_cc(i, j, k, HydroSystem<DMExpansionTest>::internalEnergy_index) = p_floor / (gamma - 1.0);
    });
}


// Particles initialization and gravitational collapse 
template <> void QuokkaSimulation<DMExpansionTest>::createInitialCICParticles()
{
	// --- Particle initialization using the Zel'dovich Approximation ---
	// 1. Start from a uniform grid (x_q) and apply a sinusoidal 1D perturbation.
	// 2. Displacement field: x = q - [D(t)/k] * sin(k*q), the 1/k factor 
	//    scales the dimensionless amplitude to physical comoving distance.
	// 3. For a EdS, the growing mode of the density contrast evolves as delta(a) = a * k * displacement. 
	//    To ensure the target a_collapse, set the initial amplitude = a_init / a_collapse.
	// 4. The peculiar velocity (v_pec) is set to match the growing mode: v_pec = a * H(a) * displacement.


	// --- Geometry ---
	// Test pass/fail logic based on the errors
	// 00. amrex::Geometry &geom information about physical dimension of the simulation and how many cells it contains; this->geom[lev] asks for data for a certain level of resolution
	
	// 1. geom.ProbLength() for the physical length of the domain (geometry.prob_lo - geometry.prob_hi); 0, 1, 2 for x, y, z
	// 2. geom.Domain() returns an amrex::Box representing the entire domain expressed in integer indices (ind_x, ind_y, ind_z), indices from 0 to amr.n_cell
	//    .length() for the number of cells per sides
	// 3. geom.ProbLo() and geom.ProbHi() physical coordinates of the lower left (Lo) and upper right (Hi) corners of the entire domain
	// 4. geom.CellSizeArray() physical size of a single cell along the three axes, calculated internally as (ProbHi - ProbLo) / number_cells.


	// Alias to access the container, based on the pointer defined in Simulation.hpp
	auto &pc = *CICParticles;   // take by ref the content of the unique_ptr CICParticles (from amrex::AmrParticleContainer), with the CIC params and methods

	// Set the base 0 level of the AMR
	const int lev = 0;

	// Geometric setup (this is pointer to the current class, QuokkaSimulation)
	const amrex::Geometry &geom = this->geom[lev];  // AMReX object for dimensions, conversion to real space from indices and periodicity
	
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> L; // physical length along the axes  (0 = x, 1 = y, 2 = z)
	L[0] = geom.ProbLength(0);
	L[1] = geom.ProbLength(1);
	L[2] = geom.ProbLength(2);

	// Or, better
	// const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
	
    // Physical quanties
	const amrex::Real G = PhysicsTraits<DMExpansionTest>::gravitational_constant;
	const amrex::Real h = PhysicsTraits<DMExpansionTest>::hubble_constant;

	const amrex::Real Mpc_to_cm = 3.08567758e24;
	const amrex::Real H0 = (h * 100.0 * 1e5) / Mpc_to_cm;   // Hubble parameter today (s^-1)
	const amrex::Real rho_crit_0 = 3.0 * H0 * H0 / (8.0 * amrex::Math::pi<amrex::Real>() * G);
	const amrex::Real rho_mean = rho_crit_0;                // comoving mean density = critical density (EdS flat universe)

	// Scale factors (priority to the inputs)
	amrex::Real a_init = PhysicsTraits<DMExpansionTest>::a_init;
	amrex::Real a_collapse = 0.5;      // target collapse moment (z=1, universe dimensions half of today)

	amrex::ParmParse pp_cosmo("cosmology");
	pp_cosmo.query("a_init", a_init);
	pp_cosmo.query("a_collapse", a_collapse);

	// H(a) for Einstein-de Sitter: H = H0 * a^(-3/2)
	const amrex::Real H_init = H0 * std::pow(a_init, -1.5);  // initial Hubble paramter for EdS from H0


	// --- Distribution of the particles along the volume and mass assignement ---

	// Number of cells of the level 0 (base) grid
	amrex::Box const& domain = geom.Domain();  // return the domain indices ( (lo_x,hi_x), (lo_y,hi_y), (lo_z,hi_z) )
	const auto dom_len = domain.length();      // number of cell along x, y, z (nx, ny, nz)

	// Set the number of particles == number of cells (1 particle per cell)
	const int n_part_x = dom_len[0];
	const int n_part_y = dom_len[1];
	const int n_part_z = dom_len[2];

	// Total particles number and mass assignement
	const amrex::Long n_part_total = static_cast<amrex::Long>(n_part_x) * n_part_y * n_part_z;  // amrex::Long to prevent overflow
	const amrex::Real total_mass   = rho_mean * L[0] * L[1] * L[2];  // inject a total mass of particles corresponding to the mean density (== critical for EdS)
	const amrex::Real particle_mass = total_mass / n_part_total;     // distribute equal mass to the parts


	amrex::Print() << "DM particles initialization...\n";
	amrex::Print() << "Mean density set: " << rho_mean << " g/cm^3\n";
	amrex::Print() << "Total DM particles created in the domain: " << n_part_total << "\n";
	amrex::Print() << "Mass per DM particle: " << particle_mass << " g\n";


	// To covert indices in physical distances
	auto prob_lo  = geom.ProbLo();         // low-left domain coordinates
	auto prob_hi  = geom.ProbHi();         // high-right domain coordinates
	const auto dx = geom.CellSizeArray();  // amrex::GpuArray<amrex::Real, 3> with the dimensions of the grid cells dx[0], dx[1], dx[2]

	// Collapse parameters: force the collapse at a_collapse
	const amrex::Real amplitude  = a_init / a_collapse;  // force the amplitude to be such that collapse happens at a_collapse
	const amrex::Real k_wave     = 2.0 * amrex::Math::pi<amrex::Real>() / L[0];  // wave vector


	// Create a particle on each cell of the grid
	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {// state_new_cc_[lev] stores hydro data at level lev
		
		// Access to the particles of the conteiner for the local tile
		const amrex::Box& tile_box = mfi.tilebox();  // amrex::Box contains integer indices of the current tile (e.g. i: 0-31, j: 0-31, k: 0-31) 
		// getParticles(lev) access all the particle in the level lev
		// make_pair() creates a pair (key, val)
		// the int mfi.index() is the global box ID, the int mfi.LocalTileIndex() is local tile ID
		// returns the amrex::ParticleTile object particles, containing the intrinsic data (Struct-of-Arrays): Positions, IDs, CPUs and the additional data (rdata and idata)
		auto& particles = pc.GetParticles(lev)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];   // map of the particles at level lev, with key [Global box, tile of the box]
	
	
		// Iterate over the 3D integer indices (i, j, k) of the grid: no particle is skipped or double-counted at tile boundaries
		amrex::Loop(tile_box, [=, &particles](int i, int j, int k) noexcept {

		// Calculate initial unperturbed Lagrangian position (q) directly from integer indices
        // prob_lo[0] + (i + 0.5) * dx[0] is the exact cell center
        amrex::Real qx = prob_lo[0] + (i + 0.5) * dx[0];  // i index is the cell left edge
        amrex::Real qy = prob_lo[1] + (j + 0.5) * dx[1];
        amrex::Real qz = prob_lo[2] + (k + 0.5) * dx[2];

		// Velocity and displacement due to the perturbation (Zel'dovich Approximation)
		// x = q - [D(t)/k] * sin(k*q); 1/k scales the dimensionless amplitude to physical comoving distance
		// For the growing mode, velocity must be in the same direction as the displacement.
		// Since we use x = q - displacement, the velocity must be -a*H*displacement.
		const amrex::Real displacement = (amplitude / k_wave) * std::sin(k_wave * qx); 
		const amrex::Real v_pec        =  a_init * H_init * displacement; 


		// --- Filling of the ParticleType struct of the particle ---

		// CIC initialization
		// ParticleType::NextID() assigns the IDs progressively on x, then y, then z: id = i_x + (i_y * nx) + (i_z * ny*nx)
		// e.g. ID = 1 (0,0,0), ID = 2 (1,0,0), ID = 3 (2, 0, 0), ..., ID = nx+1 (nx, 1, 0), ...
		using ParticleType = quokka::CICParticleContainer::ParticleType;  // create the alias ParticleType as a type for the struct of the single particles in quokka::CICParticleContainer: pos, vels, mass
		ParticleType p;   // instantiation of the stack memory space for the CIC data
		p.id()  = ParticleType::NextID();  // method that assigns uniquely a 64 bit ID to the single particle
		p.cpu() = amrex::ParallelDescriptor::MyProc(); // call to the ParallelDescriptor function: locate the processor that is currently handlig the particle

		// Comoving position: sinusoidal perturbation in x
		p.pos(0) = qx + displacement; // perturbation
		p.pos(1) = qy;
		p.pos(2) = qz;

		// Physical CIC properties: mass, vx, vy, vz
		p.rdata(quokka::CICParticleMassIdx) = particle_mass;     // mass
		p.rdata(quokka::CICParticleVxIdx)   = v_pec;		     // peculiar vx
		p.rdata(quokka::CICParticleVyIdx)   = 0.0;		         // peculiar vy
		p.rdata(quokka::CICParticleVzIdx)   = 0.0;		         // peculiar vz

		// Struct of array (SoA): puts the pos and the rdata data (mass, velocity) into separate arrays.
		particles.push_back(p);

		}); // end amrex::Loop
	} // enf for amrex::MFIter mfi

// Rearrange the particles between the processors (MPI rank) according to their position
pc.Redistribute();

amrex::Print() << "DM particles initialization completed.\n" << std::endl;

}  // end createInitialCICParticles


// Keep track of the gravitational potential
// lev, the current AMR level, dname, the name of the requested variable, mf, the MultiFab container where write the data to be saved, ncomp_cc_in, column index where insert the output data
template <> void QuokkaSimulation<DMExpansionTest>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const
{
	// compute derived variables and save in 'mf'
	if (dname == "gpot") {               // activate the function for the gravitational potential as derivate quantity
		const int ncomp = ncomp_cc_in;   // copy the target index into a local variable to allow the GPU to capture it correctly in the next lambda function
		auto const &phi_arr = phi[lev].const_arrays();  // phi[lev] contains the potential calculated by the gravity solver, const_arrays() retrives the read-only data as Array4 pointer
		auto output = mf.arrays();       // returns the pointers of writing of the output grid mf

		// Paste the gravitational potential (phi_arr) into the output container (output) as Array4 pointers
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {  // iteration over all the cells of the mf container
			// For each grid cell at the coordinate (i, j, k) of the bx box, takes the value of the gravitational potential (phi_arr) and pastes it into the corresponding cell of the output memory for the plot file (output).
			output[bx](i, j, k, ncomp) = phi_arr[bx](i, j, k); 
		});
	}
}



//  ---- Test against the analytical solution ----

// Override the (empty) hook computeAfterTimestep() in QuokkaSimulation.hpp to compare with the theory when 
// the analytical solution is valid (before the collapse).
template <> void QuokkaSimulation<DMExpansionTest>::computeAfterTimestep()
{
	// Retrieve physical parameters for the test; static variables prevent re-reading from file at every step
	static amrex::Real a_collapse  = 0.5; 
	static amrex::Real last_test_a = 0.0;   // Scale factor when the last test was performed
	static bool params_read = false;        // Flag to ensure params are read only once

	if (!params_read) {
		// Query input parameters with precedence over defaults
		amrex::ParmParse pp_cosmo("cosmology");
		pp_cosmo.query("a_collapse", a_collapse);
		params_read = true;
	}

	// Physical parameters 
	const amrex::Real G = PhysicsTraits<DMExpansionTest>::gravitational_constant;
	const amrex::Real h = PhysicsTraits<DMExpansionTest>::hubble_constant;
	const amrex::Real Mpc_to_cm = 3.08567758e24;
	const amrex::Real H0 = (h * 100.0 * 1e5) / Mpc_to_cm;   // Hubble parameter today (s^-1)
	const amrex::Real rho_crit_0 = 3.0 * H0 * H0 / (8.0 * amrex::Math::pi<amrex::Real>() * G);
	const amrex::Real rho_mean = rho_crit_0;                // Comoving mean density = critical density (EdS universe)

	// Current scale factor accessed from the simulation state
	amrex::Real a_current = this->a_now_;

	// Decimal steps (0.1, 0.2, ..., 0.8 of a_collapse) for triggering tests
	amrex::Real step_fraction = std::floor(a_current / (0.1 * a_collapse)) * 0.1;  // floor the number of tenth of a_collapse, then *0.1 to retrive the decimal scale

	// Trigger test if: 1. linear regime (step_fraction <= 0.8), 2. new interval since last test
	if (step_fraction <= 0.8 && step_fraction > (last_test_a / a_collapse + 0.01)) {   // condition to don't multiple-count and retest in the same decimal interval, + 0.01 for potential rounding errors

		amrex::Print() << "\n--- [MONITORING] a = " << a_current 
					   << " (Fraction: " << step_fraction << " of a_coll) ---\n";

		// Retrieve geometry and grid parameters (for both tests)
		const amrex::Geometry &geom = this->geom[0];    // Level 0 geometry object
		const amrex::Real L0 = geom.ProbLength(0);      // Physical extent along x-axis
		const amrex::Real dx = geom.CellSize(0);        // Grid cell size in x
		const amrex::Real dy = geom.CellSize(1);        // Grid cell size in y
		const amrex::Real dz = geom.CellSize(2);        // Grid cell size in z
		const amrex::Real prob_lo = geom.ProbLo(0);     // Lower boundary coordinate (x)
		const int nx = geom.Domain().length(0);         // Number of cells along x

		// Wave number and perturbation amplitude for Zel'dovich solution: x = q - (D(a)/k) * sin(k*q)
		// D(a) = a / a_collapse is the linear growth factor for EdS universe
		const amrex::Real k_wave = 2.0 * amrex::Math::pi<amrex::Real>() / L0;
		const amrex::Real D_factor = a_current / a_collapse;      // Linear growth factor D(a)
		const amrex::Real current_amplitude = D_factor / k_wave;  // Amplitude of sinusoidal perturbation


		// ============================================================================
		// TEST 1: Particle position error against Zel'dovich analytic solution
		// ============================================================================

		// Pointer to the CIC descriptor
		auto* descPtr = this->particleRegister_.getParticleDescriptor(quokka::ParticleType::CIC);
		
		// Structured Binding [], returnig a 3D tuple with data extracted as Struct of Arrays from all the cores with the method getParticleDataAtAllLevels()
		const auto [particle_ids, particle_real_data, particle_int_data] = descPtr->getParticleDataAtAllLevels();

		amrex::Real sum_sq_err_pos = 0.0;    // Accumulator for squared position errors
		amrex::Long n_particles = 0;         // Total particle count

		// Determin the ID offset (ID generation: particle_creation.hpp first function)
		static uint64_t id_start_global = 0;
		static bool id_start_found = false;


		if (!id_start_found && !particle_ids.empty()) {  // when ids are retrived and non replicate
			uint64_t local_min = std::numeric_limits<uint64_t>::max();
			for (std::size_t n = 0; n < particle_ids.size(); ++n) {
				uint64_t pid = static_cast<uint64_t>(particle_ids[n]);
				if (pid < local_min && pid > 0) {
					local_min = pid;
				}
			}
			// MPI reduction to synchronize the absolute minimum ID across all processors			id_start_global = local_min;
			amrex::ParallelDescriptor::ReduceRealMin(reinterpret_cast<amrex::Real&>(id_start_global));
			id_start_found = true;
		}
		
		
		// Loop over all particles and compute deviation from analytic position
		for (std::size_t n = 0; n < particle_ids.size(); ++n) {  // std::size_t n is an unsigned int, the exact object returned from .size()
			
			// Retrive the IDs and subtract the offset
			const uint64_t raw_id = static_cast<uint64_t>(particle_ids[n]);
			const uint64_t normalized_id = raw_id - id_start_global;

			// Recover the Lagrangian index i from the particle ID via modulo arithmetic
			// particle IDs are assigned sequentially: id = i_x + i_y * nx + i_z * ny*nx
			// so i is the index of the bin at x fixed where y and z varies, a sort of slice of
			// the box with ny*nz cells with the same x -> each slice with the same i feels
			// the same perturbation since the depends only on x
			const int i = static_cast<int>(normalized_id % static_cast<uint64_t>(nx));  // %: i is always 0 <= nx <= nx-1, so it is a x binning

			// Initial position of this particle a the cell center position (crucial: 1 CIC per cell, initial exactly at the center)
			const amrex::Real qx = prob_lo + (i + 0.5) * dx;  

			// Current analytical Zel'dovich position: x_analytic = q - D(a)/k * sin(k*q)
			const amrex::Real x_analytical = qx - current_amplitude * std::sin(k_wave * qx);
			
			// Retrieve numerical particle position (X coordinate) 
			const amrex::Real x_num = static_cast<amrex::Real>(particle_real_data[n][0]);
		
			// // FORCING
			// amrex::Real x_num = x_analytical; // <---------- TEST OF THE TEST

			// Compute the signed difference
			amrex::Real diff = x_num - x_analytical;

			// Apply modulo for periodic boundary conditions
			if (geom.isPeriodic(0)) {
    			diff = std::remainder(diff, L0); // Keep diff in [-L0/2, L0/2] via round to nearest integer
			} 


			// Accumulate squared error (will be normalized by particle count to get RMS)
			sum_sq_err_pos += diff * diff;
			n_particles++;
		}  // end loop over particles

		// MPI reduction to sum contributions from all processors
		amrex::ParallelDescriptor::ReduceRealSum(sum_sq_err_pos);
		amrex::ParallelDescriptor::ReduceLongSum(n_particles);

		// Compute and print RMS position error
		if (n_particles > 0) {
			amrex::Real rms_err_pos = std::sqrt(sum_sq_err_pos / n_particles);
			amrex::Print() << "  Position: RMS Error = " << rms_err_pos 
						   << " cm, Relative Error = " << (rms_err_pos / dx) << " (in units of dx)\n";
			pos_error_at_08 = rms_err_pos / dx;   // Store for final test report
		}


		// ============================================================================
		// TEST 2: CIC-deposited density error against Zel'dovich analytic solution
		// ============================================================================

		// MultiFab to store density
		amrex::MultiFab rho_cic(this->boxArray(0), this->DistributionMap(0), 1, 0);  // 1 component (density), 0 ghost cells
		rho_cic.setVal(0.0);    // set every cell in every component of the MultiFab to 0.0 (in view of the atomic sum)
		
		// Acces CIC's parameters and methods
		auto &pc = *(this->CICParticles);  // take by ref the content of the unique_ptr CICParticles (from amrex::AmrParticleContainer)


		// Deposit pointlike particle mass onto the mesh: AMReX universal function with atomic sum
		amrex::ParticleToMesh(pc,       // conteiner of all the particles
			 				  rho_cic,  // destination MultiFab where writing the data
			 			      0,        // AMR level
							  // Lambda function to take from the particle: p, the single particle; td, the ParticleTileData to access to the rdata, the id of the tile
			                  [=] AMREX_GPU_DEVICE (const auto& p, auto const& ptd, auto const& ptid) noexcept {
				return p.rdata(quokka::CICParticleMassIdx);  // return the mass of the particle
			},   // close the Lambda
			0);  // final argument: the 0 component of the destination array, rho_cic
	
		// Managing of the mass between processors
		rho_cic.SumBoundary(geom.periodicity());  // geom.periodicity() returns a boolean flag for the periodicity of the 3 directions

		// Convert mass into density (g/cm^3)
		const amrex::Real vol_cell = dx * dy * dz;
		rho_cic.mult(1.0 / vol_cell);    // AMReX GPU optimized multiplication

		// Accumulators for density error statistics (use local arrays to avoid atomic issues)
		amrex::Real sum_sq_err_rho = 0.0;
		amrex::Long n_cells_analytical = 0L;  // Count cells where solution is analytically valid. Long int

		// Loop over grid and compare numerical (CIC) vs. analytical density
		for (amrex::MFIter mfi(rho_cic); mfi.isValid(); ++mfi) {    // CPU only loop
			 // Each individual MPI process loops sequentially over the subset of sub-boxes 
   			 // assigned to it by the DistributionMapping. Inside this loop, the process 
   			 // will delegate the sub-box data (tiles) to its local CPU cores (via OpenMP) 
    		 // or to its hardware GPU threads.

			const amrex::Box& box = mfi.tilebox();         // extrct the current tile, generated by the MFIterator logical regions of indices (i,j,k) creates by MFIter
			auto const& rho_cic_arr = rho_cic.array(mfi);  // amrex::Array4<double const>, poiunter the density in position (i,j,k)

			// Local reduction using standard C++ loops (no GPU, sequentially on CPU, no atomics on stack variables)
			for (int k = box.smallEnd(2); k <= box.bigEnd(2); ++k) {  // use the box variable with the tiles indicices to limitate the threads scope
				for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
					for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {

						// Lagrangian cell position (x coordinate of cell center)
						amrex::Real qx_cell = prob_lo + (i + 0.5) * dx;

						// Linear density contrast: delta(q) = (a/a_coll) * cos(k*q)
						// Analytical density: rho(q) = rho_mean / (1 - delta)
						// Note: This is valid only when 1 - delta > 0 (before shell-crossing)
						amrex::Real delta_linear = D_factor * std::cos(k_wave * qx_cell);
						amrex::Real denom = 1.0 - delta_linear;

						// Only accumulate error if the analytical solution is valid
						// (i.e., no shell-crossing; denom remains positive and not too close to zero)
						const amrex::Real shell_crossing_threshold = 0.05;  // 5% safety margin
						if (denom > shell_crossing_threshold) {
							amrex::Real rho_analytical = rho_mean / denom;

							// Numerical density from CIC mass assignment
							// amrex::Real rho_numerical = rho_mean / denom;    // FORCE THE TEST OF THE TEST
							amrex::Real rho_numerical = rho_cic_arr(i, j, k);

							// Squared density error
							amrex::Real diff_rho = rho_numerical - rho_analytical;
							sum_sq_err_rho += diff_rho * diff_rho;
							n_cells_analytical++;
						}
					}
				}
			}
		}  // end MFIter loop

		// MPI reduction to sum contributions from all processors
		amrex::ParallelDescriptor::ReduceRealSum(sum_sq_err_rho);
		amrex::ParallelDescriptor::ReduceLongSum(n_cells_analytical);

		// Compute and print RMS density error
		if (n_cells_analytical > 0) {
			amrex::Real rms_err_rho = std::sqrt(sum_sq_err_rho / static_cast<amrex::Real>(n_cells_analytical));
			amrex::Real rms_rho_rel_err = rms_err_rho / rho_mean;   // normalize the density error to the mean (= critical here) density

			amrex::Print() << "  Density  : RMS Error = " << rms_err_rho 
						   << "  g/cm^3, Relative Error = " << rms_rho_rel_err  << " (in units of rho_mean)\n";
						   
						   // Store the error at 0.8*a_collapse for the test
			if (std::abs(step_fraction - 0.8) < 0.01) {
        		density_error_at_08 = rms_rho_rel_err;
    		}
		} else {
			amrex::Print() << "  Density  : No valid analytical cells (approaching shell-crossing)\n";
		}


		// Update the timestep to prevent re-testing the same interval
		last_test_a = a_current;
	} // end if block for decimal step trigger
}  // end of the computeAfterTimestep() override





auto problem_main() -> int
{
	// Instantiate the simulation
	QuokkaSimulation<DMExpansionTest> sim; // cretates the object sim of QuokkaSImulation, specializing the templates

	// Simulation parameters

	const amrex::Geometry &geom = sim.geom[0];
	auto domain = geom.Domain();

	amrex::Print() << "Test resolution (n_cells): "
               	   << domain.length(0) << " x " 
                   << domain.length(1) << " x " 
                   << domain.length(2) << "\n";

	// Floor hydro values to be printed
    amrex::Real rho_floor = DMExpansionTest::rho_gas_default;
    amrex::Real p_floor   = DMExpansionTest::P_gas_default;

	amrex::ParmParse pp("problem");
	pp.query("rho0_gas", rho_floor);
	pp.query("P0_gas", p_floor);
	
	amrex::Print() << "Hydro floor initialization...\n";
    amrex::Print() << "Floor density set: " << rho_floor << " g/cm^3\n";
    amrex::Print() << "Floor pressure set: " << p_floor << " g/(cm s^2)\n";
   
	
	amrex::Real a_collapse = 0.5;    // target collapse moment (z=1, universe dimensions half of today)
	amrex::Real a_init     = PhysicsTraits<DMExpansionTest>::a_init;

	amrex::ParmParse pp_cosmo("cosmology");
	pp_cosmo.query("a_init", a_init);
	pp_cosmo.query("a_collapse", a_collapse);

	// Einstein-de Sitter age: t(a) = (2/3) * (1/H0) * a^(3/2)
	const amrex::Real Mpc_to_cm = 3.08567758e24;
	const amrex::Real h = PhysicsTraits<DMExpansionTest>::hubble_constant;
	const amrex::Real H0 = (h * 100.0 * 1e5) / Mpc_to_cm;
	const amrex::Real z_collapse = (1.0 - a_collapse) / a_collapse;


	// Set the simulation duration to have the collapse
	const amrex::Real t_init = (2.0 / 3.0) * (1.0 / H0) * std::pow(a_init, 1.5);
	const amrex::Real t_collapse = (2.0 / 3.0) * (1.0 / H0) * std::pow(a_collapse, 1.5);
	sim.stopTime_ = t_collapse - t_init;
	
	// Parameters print
	amrex::Print() << "\n--- DM cosmological test parameters initialization ---" << "\n";
	amrex::Print() << "a_init = " << a_init << ", "
                   << "a_collapse = " << a_collapse << std::endl;

	amrex::Print() << "Hubble Constant (h): " << PhysicsTraits<DMExpansionTest>::hubble_constant << "\n";
	amrex::Print() << "Omega_m: " << PhysicsTraits<DMExpansionTest>::omega_m << "\n";
	amrex::Print() << "Expected simulation time (t_collapse - t_init): " << sim.stopTime_ / 3.15576e13 << "Myr" << "\n";


	// Initialization of the grid and particles
	sim.setInitialConditions();

	// Configure CIC particles to operate on the finest available refinement level
	// This ensures accurate gravity calculation for cosmological simulations
	sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::CIC)->setForceFinestLevel(true);

	amrex::Print() << "  Initial scale factor: a = " << sim.a_now_ << "\n";

	// ---- Temporal evolution from a_init to a_collapse ----
	// The loop in sim.evolve() calls computeAfterTimestep() after each substep,
	// which performs monitoring diagnostics at fixed scale factor intervals (0.1, 0.2, ..., 0.8 of a_collapse)
	sim.evolve();


	// ---- Test conclusion and pass/fail determination ----

	// Test pass/fail logic based on the errors at 0.8*a_collapse
	int status = 0;
	const amrex::Real tolerance = 0.10;    // 10% relative error tolerance
	bool pos_ok = true;
	bool rho_ok = true;

	// Position test
	if (pos_error_at_08 > tolerance || pos_error_at_08 == 0.0) {
		pos_ok = false;
		status = 1;
	}

	// Density test
	if (density_error_at_08 > tolerance || density_error_at_08 == 0.0) {
		rho_ok = false;
		status = 1;
	}

	// Syncronization of all MPI processes
	amrex::ParallelDescriptor::Barrier();

	// Final outputs
	if (amrex::ParallelDescriptor::IOProcessor()) {   // only rank 0 to avoid duplicates
		amrex::Print() << "\n" << std::string(70, '=') << "\n";
		amrex::Print() << "                    ZEL'DOVICH COSMOLOGY TEST SUMMARY                    \n";
		amrex::Print() << std::string(70, '=') << "\n";

		amrex::Print() << "\nFinal scale factor: a_now = " << sim.a_now_ << " (target = " << a_collapse << ")\n";
		amrex::Print() << "\nAnalytical validation metrics at 0.8*a_collapse:\n";
		amrex::Print() << "  Position Relative L2 Error: " << pos_error_at_08 << " (in units of cell size dx)\n";
		amrex::Print() << "  Density Relative L2 Error: " << density_error_at_08 << " (relative to rho_mean)\n";
		amrex::Print() << "  Allowed Tolerance         : " << tolerance * 100.0 << "%\n\n";

		// Position output
		if (pos_ok) {
			amrex::Print() << "[PASS] Position evolution matches Zel'dovich prediction.\n";
		} else {
			amrex::Print() << "[FAIL] Position test failed. ";
			if (pos_error_at_08 == 0.0) {
				amrex::Print() << "Reason: Diagnostic check was never triggered.\n";
			} else {
				amrex::Print() << "Reason: Error exceeds tolerance.\n";
			}
		}

		// Density output
		if (rho_ok) {
			amrex::Print() << "[PASS] Density evolution matches Zel'dovich prediction.\n";
		} else {
			amrex::Print() << "[FAIL] Density test failed. ";
			if (density_error_at_08 == 0.0) {
				amrex::Print() << "Reason: Diagnostic check was never triggered.\n";
			} else {
				amrex::Print() << "Reason: Error exceeds tolerance.\n";
			}
		}

		// FInal verdict
		if (status == 0) {
			amrex::Print() << "\n>>> TESTS PASSED <<<\n";
		} else {
			amrex::Print() << "\n>>> TEST FAILED <<<\n";
		}
		amrex::Print() << std::string(70, '=') << "\n\n";
	}

	return status;	
}





// possibile che il test fallisca per:
// ? shock numerici
// ? le posizioni oscillano perché sono discrete, mentre la soluzione analitica è continua

// Capire come imporre una risoluzione minima al test di regressione se serve per
// ritrovare la soluzione analitica con una particella per cella

// vedere se è necessario aggiugere anche la velocità analitica per il confronto con
// la soluzione numerica. Provare un confronto con velocità peculiare numerica iniziale zero


// inserire un test unitario sul mio check per capire se funziona, test attivabile con un flag
// boolenano del tipo if debug=true;:

// All'interno del tuo test, prima della validazione L2
// if (check_analytical_code) {
//     auto& ptile = sim.particleData_.GetParticles(0); // Livello 0
//     const real_t a_target = 0.8 * a_collapse;
//     const real_t L = sim.geom[0].ProbLength(0); // Lunghezza box in X

//     for (auto& p : ptile) {
//         // q è la posizione iniziale (Lagrangiana) che devi aver salvato o ricostruito
//         // Se le particelle erano in una griglia perfetta all'inizio:
//         real_t q_x = p.pos(0); // Attenzione: questo presuppone che p.pos(0) sia q_x
        
//         // Applica lo spostamento di Zel'dovich
//         real_t displacement = (a_target / a_collapse) * (L / (2.0 * M_PI)) * std::sin(2.0 * M_PI * q_x / L);
//         p.pos(0) = q_x - displacement; 
        
//         // Opzionale: azzera le velocità se il test le controlla
//         p.vel(0) = 0.0; 
//     }
// }


// vedere se fare file di input distinti per il test e per la simulazione fisica



// Forzare in qualche modo il salvataggio del potenziale gravitazionale

// stampare anche l'errore massimo (L-infinity norm) oltre all'RMS, per individuare 
// eventuali particelle "impazzite" ai bordi dei processori.

// aggiungere codice in cui testo che il codice sia giusto facendo il test con uno snapshot 
// campionato esattamente dalla soluzione analitica di Zel'dovich per le posizioni


// vedere se tutti i parametri geometrici che ho riscritto nel problem_main() per
// la soluzione analitica sono necessari

// vedere se con il nuovo solver integrato 	static constexpr double cosmology_dt_limit = 0.01; // according to the default
// nei Traits ha ancora senso

// per isolare meglio la fisica, provare anche il test nel caso di gamma = 1
// eds isoterma

// valutare se un floor di pressione e densità troppo basso può
// influenzare il test (Con densità così basse, piccole oscillazioni numeriche
// causate dal potenziale gravitazionale delle particelle possono
// produrre pressioni o energie negative. La velocità del suono 
// potrebbe dare un valore negativo -> crash con un errore aritmetico.)
// const Real rho_floor = 1.0e-20; // Almeno qualche ordine di grandezza in più
// const Real p_floor   = 1.0e-25;

// vedere se renderlo test di regressione

// mettere il parser dei parametri
// fare file di input, vedere se mettere input del solver gt, magari rendere parsable a_collapse,
// comoving_mean_density

// mettere risoluzione maggiore in x che in y e z. Magari modificare anche con meno particelle
// lungo y e z: n_particles_1d -> n_particles_x, L -> Lx, Ly, Lz, ...

// aggiungere sigma8 ai parametri della struct in Cosmology.hpp

// implementare il tempo di free falling, che una particella impiega per cadere al centro della perturbazione
// se non ci fosse espansione

// mettere print utili in fase di svolgimento simulazione


// più avanti, magari valutare se mettere 
// static bool do_split_particles = false; // NOLINT
// static int split_factor = 8;		// NOLINT
// per dividere in 8 CIC più leggere una CIC in presenza di AMR, per evitare di avere troppo
// poche particelle in celle che diventano troppo piccole, e quiandi avrei rumore di conteggio,
// shot noise

// vedere se aggiungere le CIC come nella logica di BinaryOrbitCIC in cui tra gli input ho file di testo
// ASCII tramite InitFromAsciiFile (per ora descriptor per accedere a elem container):
// // 2
// 3.125e12 0. 0. 2.0e34 0. 10332860. 0.
// -3.125e12 0. 0. 2.0e34 0. -10332860. 0.
// e si usa 
// template <> void QuokkaSimulation<BinaryOrbit>::createInitialCICParticles()
// {
// 	// read particles from ASCII file
// 	const int nreal_extra = 4; // mass vx vy vz
// 	CICParticles->SetVerbose(1);
// 	CICParticles->InitFromAsciiFile("../inputs/BinaryOrbit_particles.txt", nreal_extra, nullptr);

// 	// test particle splitting
// 	// (this is intended to only be used when restarting at a higher resolution)
// 	if (do_split_particles) {
// 		amrex::Print() << "Splitting CICParticles using split_factor = " << split_factor << "\n";
// 		int const lev = 0; // all CICParticles are on level 0
// 		particleRegister_.getParticleDescriptor(quokka::ParticleType::CIC)->splitParticles(lev, split_factor);
// 	}
// }


// introdurre AMR (se ci sarà una fisica che lo giustifica)

// assegnare una velocità peculiare iniziale legata allo spostamento tramite il tasso di crescita lineare 
// Senza velocità iniziale, la perturbazione impiegherà più tempo a crescere. Fare una documentazione della
// soluzione analitica del pancake e del motivo della vpec iniziale

// magari mettere bounds sulla tolleranza rispetto a un riferimento /sol an se possibile

// magari, se il test funziona, pensare se aggiungere hydro, il gas, con tutti
// i suoi gamma, press, densità, ... 
// OSS: in simulation.hpp righe circa 950 ci sono density, temperature, ... floor
// vedere se tornano utili e eleganti per questo test

// vedere nella cartella src/problems il file /data/mfulghieri/quokka/src/problems/ProblemHelpers.cmake
// per rendere CosmologicalExpansion un test

// poi estendere il collasso da 1d a 3d

// Magari Zel'dovich comolessivo con hydro + CIC

// capire come funziona il Poisson solver

// aggiungere stampa periodica del numero delle statistiche delle particelle come 
// prima del problem:main in BinaryOrbitCIC

// vedere se aggiungere le particles tracers (vedi prime 1000 righe di simulation.hpp)
// per tracciare


// Vedere se va reso compatibile con GPU la creazione delle particles quando si fa
// il pushback


// OBS: Griglia Euleriana: lo spazio è diviso in celle fisse. Calcoli come le quantità (massa, momento) fluiscono da una cella all'altra.
//      Griglia Lagrangiana: le celle si muovono con il fluido.
// Quokka è un codice IBRIDO: idrodinamica (Gas): Usa una griglia Euleriana (AMR). Il gas "scorre" attraverso le celle fisse.
// Materia Oscura (Particelle): Usa un approccio Lagrangiano. Le particelle si muovono liberamente nello spazio seguendo le equazioni del moto.
// L'unione (CIC): Il metodo Cloud-in-Cell (CIC) che stai usando è il "ponte": proietta la massa lagrangiana delle particelle sulla griglia euleriana per calcolare il potenziale gravitazionale (Poisson solver)._self_gravity_enabled = true;
	