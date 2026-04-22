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

#include <utility>        // for std::make_pairconst amrex::Real total_mass = rho_mean * (L * L * L);
#include <cmath>          // for std::sin, std::floor



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

	// Distribution of the particles along the volume and mass assignement
	const int n_part_1d             = 64;                     // number of CICs
	const amrex::Real dx_particles  = L / n_part_1d;          // x separation between each particle
	const amrex::Real total_mass    = rho_mean * (L * L * L); // inject a total mass of particles corresponding to the mean density (== critical for EdS)

	const amrex::Long n_part_total  = static_cast<amrex::Long>(n_part_1d) * n_part_1d * n_part_1d; // amrex::Long to prevent overflow
	const amrex::Real particle_mass = total_mass / n_part_total; // distributing equal mass to the parts


	amrex::Print() << "DM particles initialization...\n";
	amrex::Print() << "Mean density set: " << rho_mean << " g/cm^3\n";
	amrex::Print() << "Total DM particles created in the domain: " << n_part_total << "\n";
	amrex::Print() << "Mass per DM particle: " << particle_mass << " g\n";


	// To covert indices in physical distances
	auto prob_lo  = geom.ProbLo();         // low-left domain coordinates
	auto prob_hi  = geom.ProbHi();         // high-right domain coordinates
	const auto dx = geom.CellSizeArray();  // amrex::GpuArray<amrex::Real, 3> with the dimensions of the grid cells dx[0], dx[1], dx[2]

	// Collapse parameters: force the collapse at a_collapse
	const amrex::Real a_collapse = 0.5;                  // target collapse moment (z=1, universe dimensions half of today)
	const amrex::Real amplitude  = a_init / a_collapse;  // force the amplitude to be such that collapse happens at a_collapse
	const amrex::Real k_wave     = 2.0 * amrex::Math::pi<amrex::Real>() / L;  // wave vector


	// Fictious particle grid for the initial unperturbed, to avoid 3 nested for
	amrex::IntVect part_lo(0, 0, 0);                                     // origin of the fictious particle grid
	amrex::IntVect part_hi(n_part_1d - 1, n_part_1d - 1, n_part_1d - 1); // -1 since the indices start from 0
	amrex::Box part_grid_full(part_lo, part_hi);                         // rectangular region in the indices space, stores the range of the available indices


	// Compute how many CICs are in the tile AMR
	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) { // state_new_cc_[lev] stores hydro data at level lev
	// Iteration over all the boxes assigned to current processor

		// Access to the particles of the conteiner for the local tile
		auto &particles = pc.GetParticles(lev)[std::make_pair(mfi.index(), mfi.LocalTileIndex())]; // map of the particles at level lev, with key [Global box, tile of the box]
		const amrex::Box& tile_box = mfi.tilebox();  // amrex::Box contains integer indices of the current calculation region (e.g. i: 0-31, j: 0-31, k: 0-31) 

		// Physical borders of the current tile
		amrex::Real x_min = prob_lo[0] + tile_box.smallEnd(0) * dx[0];     // smallEnd(0) for the inclusive lower bound in x (0)
		amrex::Real x_max = prob_lo[0] + (tile_box.bigEnd(0) + 1) * dx[0]; // bigEnd for the inclusive upper bound, +1 to include the right border of the last cell

   		amrex::Real y_min = prob_lo[1] + tile_box.smallEnd(1) * dx[1];
   		amrex::Real y_max = prob_lo[1] + (tile_box.bigEnd(1) + 1) * dx[1];

   		amrex::Real z_min = prob_lo[2] + tile_box.smallEnd(2) * dx[2];
   		amrex::Real z_max = prob_lo[2] + (tile_box.bigEnd(2) + 1) * dx[2];


		// Conversion of the cell indices in spatial coordinates for the begin and end of the tile
		amrex::IntVect p_tile_lo(//small offset (1e-10) for numerical precision at borders
       	static_cast<int>(std::floor((x_min - prob_lo[0] + 1e-10) / dx_particles)), // floor rounds to the closest smaller integer
       	static_cast<int>(std::floor((y_min - prob_lo[1] + 1e-10) / dx_particles)),
       	static_cast<int>(std::floor((z_min - prob_lo[2] + 1e-10) / dx_particles))
   		);

		// Find the indices of the CICs that falls in this x, y, z range: index = position / spacing
   		amrex::IntVect p_tile_hi(
       	static_cast<int>(std::floor((x_max - prob_lo[0] - 1e-10) / dx_particles)),
       	static_cast<int>(std::floor((y_max - prob_lo[1] - 1e-10) / dx_particles)),
       	static_cast<int>(std::floor((z_max - prob_lo[2] - 1e-10) / dx_particles))
   		);

		// Intersection: a new box with only the indies both in part_grid_full and the current processor tile: the first is the physical limit, preventing local processor generate unphysical particles
		amrex::Box local_part_box = amrex::Box(p_tile_lo, p_tile_hi) & part_grid_full;  // amrex::Box(p_tile_lo, p_tile_hi): temporary grid with physical space handled by the processor


		// amrex::Loop(tile_box,...) = 3 nested loops of the shape: for (int k = lo.z; k <= hi.z; ++k)
		if (local_part_box.ok()) { // proceed only if there are particles in this tile (.ok() is true if at least one point, ie intersection successful)
			amrex::Loop(local_part_box, [=, &particles](int i, int j, int k) noexcept { // capure by ref the conteiner &particles, with current index of the particle grid (i,j,k)
				
				// Unperturbed initial lagrangian position (center of the particle cell)
				amrex::Real qx = (i + 0.5) * dx_particles;   // i index is the cell left edge
           		amrex::Real qy = (j + 0.5) * dx_particles;
           		amrex::Real qz = (k + 0.5) * dx_particles;


				// Velocity and displacement due to the perturbation (Zel'dovich Approximation)
				//  x = q - [D(t)/k] * sin(k*q); 1/k scales the dimensionless amplitude to physical comoving distance
				const amrex::Real displacement = (amplitude / k_wave) * std::sin(k_wave * qx); //  x = q - [D(t)/k] * sin(k*q)
				const amrex::Real v_pec        = a_init * H_init * displacement;               // contrast the decay mode, set to match the growing mode


				// CIC initialization
				using ParticleType = quokka::CICParticleContainer::ParticleType;  // create the alias ParticleType as a type for the struct of the single particles in quokka::CICParticleContainer: pos, vels, mass
				ParticleType p;   // instantiation of the stack memory space for the CIC data
				p.id() = ParticleType::NextID();  // method that assigns uniquely a 64 bit ID to the single particle
				p.cpu() = amrex::ParallelDescriptor::MyProc(); // call to the ParallelDescriptor function: locate the processor that is currently handlig the particle


				// Sinusoidal perturbation in x
				p.pos(0) = qx - displacement; // perturbation
				p.pos(1) = qy;
				p.pos(2) = qz;

				// Physical CIC properties: mass, vx, vy, vz
				p.rdata(quokka::CICParticleMassIdx) = particle_mass;     // mass
				p.rdata(quokka::CICParticleVxIdx)   = v_pec;		     // peculiar vx
				p.rdata(quokka::CICParticleVyIdx)   = 0.0;		         // peculiar vy
				p.rdata(quokka::CICParticleVzIdx)   = 0.0;		         // peculiar vz

				particles.push_back(p);

			}); // end amrex::Loop
		} // end if(local_part_box.ok())
	} // end MFIter

	// Rearrange the particles between the processors (MPI rank) according to their position
	pc.Redistribute();

	amrex::Print() << "DM particles initialization completed.\n" << std::endl;
}




auto problem_main() -> int
{
	// Instantiate the simulation
	QuokkaSimulation<DMExpansionTest> sim; // cretates the object sim of QuokkaSImulation, specializing the templates

	// Simulation parameters
	const amrex::Real a_collapse = 0.5;    // target collapse moment (z=1, universe dimensions half of today)
	const amrex::Real z_collapse = (1.0 - a_collapse) / a_collapse;
	const amrex::Real a_init = PhysicsTraits<DMExpansionTest>::a_init;

	// Einstein-de Sitter age: t(a) = (2/3) * (1/H0) * a^(3/2)
	const amrex::Real Mpc_to_cm = 3.08567758e24;
	const amrex::Real h = PhysicsTraits<DMExpansionTest>::hubble_constant;
	const amrex::Real H0 = (h * 100.0 * 1e5) / Mpc_to_cm;

	// Set the simulation duration to have the collapse
	const amrex::Real t_init = (2.0 / 3.0) * (1.0 / H0) * std::pow(a_init, 1.5);
	const amrex::Real t_collapse = (2.0 / 3.0) * (1.0 / H0) * std::pow(a_collapse, 1.5);
	sim.stopTime_ = t_collapse - t_init;
	
	// Parameters print
	amrex::Print() << "\n--- DM cosmological test parameters ---" << "\n";
	amrex::Print() << "Hubble Constant (h): " << PhysicsTraits<DMExpansionTest>::hubble_constant << "\n";
	amrex::Print() << "Omega_m: " << PhysicsTraits<DMExpansionTest>::omega_m << "\n";
	amrex::Print() << "Initial scale factor (a_init): " << PhysicsTraits<DMExpansionTest>::a_init << "\n";
	amrex::Print() << "Expected simulation time (t_collapse - t_init): " << sim.stopTime_ << "\n";

	// Initialization
	sim.setInitialConditions();

	// Temporal evolution
	sim.evolve();

	amrex::Print() << "\nCIC + cosmology Results:\n";
	amrex::Print() << "  Final a = " << sim.a_now_ << " (expected " << a_collapse << ")\n";

	return 0;
}





// mettere il parser dei parametri
// fare file di input, vedere se mettere input del solver gt, magari rendere parsable a_collapse,
// comoving_mean_density

// mettere risoluzione maggiore in x che in y e z. Magari modificare anche con meno particelle
// lungo y e z: n_particles_1d -> n_particles_x, L -> Lx, Ly, Lz, ...

// analisi yt e python

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

// vedere nella cartella src/problems il file /data/mfulghieri/quokka/src/problems/ProblemHelpers.cmake
// per rendere CosmologicalExpansion un test

// poi estendere il collasso da 1d a 3d


// OBS: Griglia Euleriana: lo spazio è diviso in celle fisse. Calcoli come le quantità (massa, momento) fluiscono da una cella all'altra.
//      Griglia Lagrangiana: le celle si muovono con il fluido.
// Quokka è un codice IBRIDO: idrodinamica (Gas): Usa una griglia Euleriana (AMR). Il gas "scorre" attraverso le celle fisse.
// Materia Oscura (Particelle): Usa un approccio Lagrangiano. Le particelle si muovono liberamente nello spazio seguendo le equazioni del moto.
// L'unione (CIC): Il metodo Cloud-in-Cell (CIC) che stai usando è il "ponte": proietta la massa lagrangiana delle particelle sulla griglia euleriana per calcolare il potenziale gravitazionale (Poisson solver).