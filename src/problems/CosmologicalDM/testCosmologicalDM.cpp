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

#include <utility>        // for std::make_pairconst 
#include <cmath>          // for std::sin, std::floor



// Struct tag for the templates, with the defalt hydro values
struct DMExpansionTest{
	static constexpr amrex::Real rho_gas_default = 1.0e-30;    // low density
	static constexpr amrex::Real P_gas_default   = 1.0e-40;    // low pressure (enough to have low sound speed and thus small dt)
};

template <> struct quokka::EOS_Traits<DMExpansionTest> {
	static constexpr amrex::Real gamma = 5.0 / 3.0;
	static constexpr amrex::Real mean_molecular_weight = C::m_u;
};

// Traits: physics and particles
// Introduction of CIC from particle_types.hpp
template <> struct Particle_Traits<DMExpansionTest>{
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

	// Initial conditions: precedence to user-entered values (override if present in the .in file)
	amrex::ParmParse pp("problem");
	pp.query("rho0_gas", rho_floor);
	pp.query("P0_gas", p_floor);
	pp.query("gamma", gamma);

    amrex::Print() << "Grid initialization: setting floor density and pressure...\n";

    const amrex::Box &indexRange = grid_elem.indexRange_;       // set of the indices of the grid patch (e.g. from 0 to 31 in x, y, z)
    const amrex::Array4<double> & state_cc = grid_elem.array_;  // Array4 is a pointer to the data

    // Parallel loop over the spatial indices on GPU
    amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
        // Small floor placeholder value (avoid /0 in vel but not affect gravity)
        state_cc(i, j, k, HydroSystem<DMExpansionTest>::density_index) = rho_floor; 
        state_cc(i, j, k, HydroSystem<DMExpansionTest>::x1Momentum_index) = 0;
        state_cc(i, j, k, HydroSystem<DMExpansionTest>::x2Momentum_index) = 0;
        state_cc(i, j, k, HydroSystem<DMExpansionTest>::x3Momentum_index) = 0;
        state_cc(i, j, k, HydroSystem<DMExpansionTest>::energy_index) = p_floor / (gamma - 1.0);
        state_cc(i, j, k, HydroSystem<DMExpansionTest>::internalEnergy_index) = p_floor / (gamma - 1.0);
    });
}


// Particles initialization and gravitational collapse (https://amrex-codes.github.io/amrex/docs_html/Particle.html)
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
	const amrex::Geometry &geom = this->geom[lev];  // AMReX object for dimensions, conversion to real space from indices and periodicity
	
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> L; // physical length along the axes  (0 = x, 1 = y, 2 = z)
	L[0] = geom.ProbLength(0);
	L[1] = geom.ProbLength(1);
	L[2] = geom.ProbLength(2);

	
    // Physical quanties
	const amrex::Real G = PhysicsTraits<DMExpansionTest>::gravitational_constant;
	const amrex::Real h = PhysicsTraits<DMExpansionTest>::hubble_constant;

	const amrex::Real Mpc_to_cm = 3.08567758e24;
	const amrex::Real H0 = (h * 100.0 * 1e5) / Mpc_to_cm;   // Hubble parameter today (s^-1)
	const amrex::Real rho_crit_0 = 3.0 * H0 * H0 / (8.0 * amrex::Math::pi<amrex::Real>() * G);
	const amrex::Real rho_mean = rho_crit_0;                // comoving mean density = critical density (EdS flat universe)

	// H(a) for Einstein-de Sitter: H = H0 * a^(-3/2)
	const amrex::Real a_init = PhysicsTraits<DMExpansionTest>::a_init;
	const amrex::Real H_init = H0 * std::pow(a_init, -1.5);  // initial Hubble paramter for EdS from H0

	// Distribution of the particles along the volume and mass assignement

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
	const amrex::Real a_collapse = 0.5;                  // target collapse moment (z=1, universe dimensions half of today)
	const amrex::Real amplitude  = a_init / a_collapse;  // force the amplitude to be such that collapse happens at a_collapse
	const amrex::Real k_wave     = 2.0 * amrex::Math::pi<amrex::Real>() / L[0];  // wave vector


	// Compute how many CICs are in the tile AMR
	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {// state_new_cc_[lev] stores hydro data at level lev
		
		// Access to the particles of the conteiner for the local tile
		const amrex::Box& tile_box = mfi.tilebox();  // amrex::Box contains integer indices of the current tile (e.g. i: 0-31, j: 0-31, k: 0-31) 
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
		const amrex::Real displacement = - (amplitude / k_wave) * std::sin(k_wave * qx); 
		const amrex::Real v_pec        = - a_init * H_init * displacement; 

		// CIC initialization
		using ParticleType = quokka::CICParticleContainer::ParticleType;  // create the alias ParticleType as a type for the struct of the single particles in quokka::CICParticleContainer: pos, vels, mass
		ParticleType p;   // instantiation of the stack memory space for the CIC data
		p.id() = ParticleType::NextID();  // method that assigns uniquely a 64 bit ID to the single particle
		p.cpu() = amrex::ParallelDescriptor::MyProc(); // call to the ParallelDescriptor function: locate the processor that is currently handlig the particle

		// Comoving position: sinusoidal perturbation in x
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

	// Retrive the descriptor for the CIC (via getParticleDescriptor), processing the particles on the finest available level
	sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::CIC)->setForceFinestLevel(true);

	// Temporal evolution
	sim.evolve();


	// ---- Check against the analytical solution ----

	amrex::Print() << "\n Start testing against the anlytical solution (Zel'dovich approximation)...\n";

	// Get the number of particles
	const int n_particles = sim.particleRegister_.getParticleDescriptor(quokka::ParticleType::CIC)->getNumParticles();
	amrex::Print() << " Total DM particles = " << n_particles;


	// Variable for the status of the test
	int status = 0;   //  0 = Pass, 1 = Fail  



	////////// Proseguire //////////
	// calcolo posizioni analitiche
	


	amrex::Print() << "\nCIC + cosmology Results:\n";
	amrex::Print() << "  Final a = " << sim.a_now_ << " (expected " << a_collapse << ")\n";

	return 0;
}






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


// OBS: Griglia Euleriana: lo spazio è diviso in celle fisse. Calcoli come le quantità (massa, momento) fluiscono da una cella all'altra.
//      Griglia Lagrangiana: le celle si muovono con il fluido.
// Quokka è un codice IBRIDO: idrodinamica (Gas): Usa una griglia Euleriana (AMR). Il gas "scorre" attraverso le celle fisse.
// Materia Oscura (Particelle): Usa un approccio Lagrangiano. Le particelle si muovono liberamente nello spazio seguendo le equazioni del moto.
// L'unione (CIC): Il metodo Cloud-in-Cell (CIC) che stai usando è il "ponte": proietta la massa lagrangiana delle particelle sulla griglia euleriana per calcolare il potenziale gravitazionale (Poisson solver)._self_gravity_enabled = true;
	