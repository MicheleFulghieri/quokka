//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testCosmologicalDarkMatter.cpp
/// \brief Defines a test problem including cosmology and dark matter
///

#include "QuokkaSimulation.hpp"
#include "particles/particle_types.hpp"
#include "cosmology/Cosmology.hpp"

#include <AMReX_Math.H>   // for pi

#include <utility>        // for std::make_pair
#include <cmath>          // for std::sin



// Empty struct, tag for the templates
struct DMExpansionTest{
};

template <> struct quokka::EOS_Traits<DMExpansionTest> {
	static constexpr double gamma = 5.0 / 3.0;
	static constexpr double mean_molecular_weight = C::m_u;
};

// Traits: physics and particles
// Introduction of CIC from particle_types.hpp
template <> struct Particle_Traits<DMExpansionTest>{
    static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC;
};

// Physics traits
template <> struct Physics_Traits<DMExpansionTest> {
    static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_cosmology_enabled = true;
	static constexpr bool is_self_gravity_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = 0;
	static constexpr bool is_dust_enabled = false;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;

// Cosmology parameters
	static constexpr double omega_m = 1.0;
	static constexpr double omega_r = 0.0;
	static constexpr double omega_lambda = 0.0;
	static constexpr double hubble_constant = 0.7;	   // h = 0.7 (H0 = 70 km/s/Mpc)
	static constexpr double a_init = 0.01;		       // start at z = 99
	static constexpr double cosmology_dt_limit = 0.01; // according to the default
};


// Grid initialization (only background, no hydro solver enabled, only the floor state_cc)
template <> void QuokkaSimulation<DMExpansionTest>::setInitialConditionsOnGrid(quokka::grid const &grid_elem) {

    const amrex::Box &indexRange = grid_elem.indexRange_;       // set of the indices of the grid patch (e.g. from 0 to 31 in x, y, z)
    const amrex::Array4<double> & state_cc = grid_elem.array_;  // Array4 is a pointer to the data

    // Floor hydro values to avoid nan 
    const Real rho_floor = 1.0e-30;
    const Real p_floor = 1.0e-35;
    const amrex::Real gamma = quokka::EOS_Traits<DMExpansionTest>::gamma;
    amrex::Print() << "Grid initialization: setting floor density to 1.0e-30\n";

    // Parallel loop over the spatial indices on GPU
    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
        // Small floor placeholder value (avoid /0 in vel but not affect gravity)
        state_cc(i, j, k, HydroSystem<DMExpansionTest>::density_index) = 1.0e-30; 
        state_cc(i, j, k, HydroSystem<DMExpansionTest>::x1Momentum_index) = rho_floor;
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
	// 2. The displacement field is defined as: x = q - [D(t)/k] * sin(k*q), where the 1/k factor 
	//    scales the dimensionless amplitude to physical comoving distance.
	// 3. For an Einstein-de Sitter (EdS) universe, the growing mode of the density contrast 
	//    evolves as delta(a) = a * k * displacement. To ensure structure formation at the 
	//    target a_collapse, set the initial amplitude = a_init / a_collapse.
	// 4. The peculiar velocity (v_pec) is set to match the growing mode: v_pec = a * H(a) * displacement.
	//    This ensures that the particles possess the correct initial momentum to overcome 
	//    Hubble expansion and collapse into a "pancake" at the predicted time.

	// Descriptor to access the container
	auto &pc = *CICParticles;   // takes by ref the content of the unique_ptr CICParticles (from amrex::AmrParticleContainer), with the CIC params and methods

	// Set the base 0 level of the AMR
	const int lev = 0;

	// Geometric setup (this is pointer to the current class, QuokkaSimulation)
	const amrex::Geometry &geom = this->geom[lev]; // AMReX object for dimensions, conversion to real space from indices and periodicity
	const amrex::Real L = geom.ProbLength(0);      // physical length along the 0 ax  (0 = x, 1 = y, 2 = z)
	
    // Physical quanties
    const amrex::Real Mpc_to_cm = 3.08567758e24;
	const amrex::Real h = PhysicsTraits<DMExpansionTest>::hubble_constant;
	const amrex::Real H0 = (h * 100.0 * 1e5) / Mpc_to_cm;   // Hubble parameter today (s^-1)
	const amrex::Real G = PhysicsTraits<DMExpansionTest>::gravitational_constant;
	const amrex::Real rho_crit_0 = 3.0 * H0 * H0 / (8.0 * amrex::Math::pi<amrex::Real>() * G);
	const amrex::Real rho_mean = rho_crit_0;                // comoving mean density = critical density (EdS flat universe)

	// H(a) for Einstein-de Sitter: H = H0 * a^(-3/2)
	const amrex::Real a_init = PhysicsTraits<DMExpansionTest>::a_init;
	const amrex::Real H_init = H0 * std::pow(a_init, -1.5);  // initial Hubble paramter for EdS from H0

	// Cell volume and CIC mass
	const amrex::Real *dx = geom.CellSize();	        // pointer to the first elem of an array containg the cell dimensions in x, y, z
	const amrex::Real cell_vol = dx[0] * dx[1] * dx[2]; // cell volume computed by hands
	const amrex::Real particle_mass = rho_mean * cell_vol;

	amrex::Print() << "DM particles initialization...\n";
	amrex::Print() << "Mean density set: " << rho_mean << " g/cm^3\n";
	amrex::Print() << "Cell volume: " << cell_vol << " cm^3\n";
	amrex::Print() << "Mass per particle: " << particle_mass << " g\n";


	// Collapse parameters: force the collapse at a_collapse
	const amrex::Real a_collapse = 0.5;                  // target collapse moment (z=1, universe dimensions half of today)
	const amrex::Real amplitude  = a_init / a_collapse;  // force the amplitude to be such that collapse happens at a_collapse
	const amrex::Real k_wave     = 2.0 * amrex::Math::pi<amrex::Real>() / L;  // wave vector

	// Iteration on the particles and perturbation (for since particles.push_back(p) not amrex::ParallelFor-safe)
	amrex::Long local_p_count = 0; // counter of the particles creates by this single processor


	// MultiFab Iterator for MPI
	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		// Acess to the particles of the tile
		auto &particles = pc.GetParticles(lev)[std::make_pair(mfi.index(), mfi.LocalTileIndex())]; // map with the two keys [box, tile (subdivision of the box)]
		const amrex::Box &tile_box = mfi.tilebox(); // amrex::Box set of index (i,j,k), eg i=0,...31, tilebox() gives the current working borders

		// amrex::Loop(tile_box,...) = 3 nested loops of the shape: for (int k = lo.z; k <= hi.z; ++k)
		amrex::Loop(tile_box, [=, &particles, &local_p_count](int i, int j, int k) {
			// Cell-centered  CIC (comoving) positions
			amrex::Real x = (i + 0.5) * dx[0]; // i index is the cell left edge
			amrex::Real y = (j + 0.5) * dx[1];
			amrex::Real z = (k + 0.5) * dx[2];

			// Velocity and displacement due to the perturbation (Zel'dovich Approximation)
			//  x = q - [D(t)/k] * sin(k*q); 1/k scales the dimensionless amplitude to physical comoving distance
			const amrex::Real displacement = (amplitude / k_wave) * std::sin(k_wave * x); //  x = q - [D(t)/k] * sin(k*q)
			const amrex::Real v_pec        = a_init * H_init * displacement;              // contrast the decay mode, set to match the growing mode

			// CIC initialization
			using ParticleType = quokka::CICParticleContainer::ParticleType;  // create the alias ParticleType as a type for the struct of the single particles in quokka::CICParticleContainer: pos, vels, mass
			ParticleType p;   // instantiation of the stack memory space for the CIC data
			p.id() = ParticleType::NextID();  // method that assigns uniquely a 64 bit ID to the single particle
			p.cpu() = amrex::ParallelDescriptor::MyProc(); // call to the ParallelDescriptor function: locate the processor that is currently handlig the particle

			// Sinusoidal perturbation in x
			p.pos(0) = x - displacement; // perturbation
			p.pos(1) = y;
			p.pos(2) = z;

			// Physical CIC properties: mass, vx, vy, vz
			p.rdata(quokka::CICParticleMassIdx) = particle_mass; // mass
			p.rdata(quokka::CICParticleVxIdx) = v_pec;		     // peculiar vx
			p.rdata(quokka::CICParticleVyIdx) = 0.0;		     // peculiar vy
			p.rdata(quokka::CICParticleVzIdx) = 0.0;		     // peculiar vz

			particles.push_back(p);
			local_p_count++; // update the particle counter of this processor
		});
	}

	// Rearrange the particles between the processors (MPI rank) according to their position
	pc.Redistribute();
	
	amrex::Long global_p_count = local_p_count;
	amrex::ParallelDescriptor::ReduceLongSum(global_p_count); // ReduceLongSum: sum the values of each processor

	amrex::Print() << "Total DM particles created in the domain: " << global_p_count << "\n\n";
}

auto problem_main() -> int
{
	// Instantiate the simulation
	QuokkaSimulation<DMExpansionTest> sim; // cretates the object sim of QuokkaSImulation, specializing the templates

	// Parameters print
	amrex::Print() << "\n--- DM cosmological test parameters ---" << "\n";
	amrex::Print() << "Hubble Constant (h): " << PhysicsTraits<DMExpansionTest>::hubble_constant << "\n";
	amrex::Print() << "Omega_m: " << PhysicsTraits<DMExpansionTest>::omega_m << "\n";
	amrex::Print() << "Initial scale factor (a_init): " << PhysicsTraits<DMExpansionTest>::a_init << "\n";

	// Initialization
	sim.setInitialConditions();

	// Set simulation parameters
	const amrex::Real yr_to_s = 3.15576e7;
	sim.stopTime_             = 1.0e8 * yr_to_s; // 100 Myr
	sim.maxTimesteps_         = 100000;
	sim.cflNumber_            = 0.3;

	// Allow overrides from input file
	amrex::ParmParse pp_amr("amr");
	pp_amr.query("max_timesteps", sim.maxTimesteps_);
	if (pp_amr.query("stop_time", sim.stopTime_)) {
		sim.stopTime_ *= yr_to_s;
	}
	amrex::ParmParse pp_quokka("quokka");
	pp_quokka.query("cfl", sim.cflNumber_);

	// Temporal evolution
	sim.evolve();

	const amrex::Real a_collapse = 0.5;    // for the print: target collapse moment (z=1, universe dimensions half of today)
	amrex::Print() << "\nCIC + cosmology Results:\n";
	amrex::Print() << "  Final a = " << sim.a_now_ << " (expected " << a_collapse << ")\n";

	return 0;
}


// in ogni caso, perché questo problema abbia un senso, occorre sviluppare la cosmologia per le particles

// problemi nel leggere a_init nel parser

// amrex::Math::pi<amrex::Real>() al posto di pigreco con #include <AMReX_Math.H>   // for pi

// magari rendere parsable a_collapse

// fare file di input, vedere se mettere input del solver gt

// implementare il tempo di free falling, che una particella impiega per cadere al centro della perturbazione
// se non ci fosse espansione

// mettere print utili in fase di svolgimento simulazione

// mettere il parser dei parametri

// comoving_mean_density nel file di input

// verificare se la cosmologia si applica anche alle particelle e alle CIC e eventualmente come

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

// vedere se aggiungere velocità peculiari alle CIC e come influenzano il collasso (se ha senso)


// assegnare una velocità peculiare iniziale legata allo spostamento tramite il tasso di crescita lineare 
// Senza velocità iniziale, la perturbazione impiegherà più tempo a crescere. Fare una documentazione della
// soluzione analitica del pancake e del motivo della vpec iniziale

// magari mettere bounds sulla tolleranza rispetto a un riferimento /sol an se possibile

// magari, se il test funziona, pensare se aggiungere hydro, il gas, con tutti
// i suoi gamma, press, densità, ...


// vedere nella cartella src/problems il file /data/mfulghieri/quokka/src/problems/ProblemHelpers.cmake
// per rendere CosmologicalExpansion un test