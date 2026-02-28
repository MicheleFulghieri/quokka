#ifndef PHYSICS_INFO_HPP_ // NOLINT
#define PHYSICS_INFO_HPP_

#include "AMReX_REAL.H"
#include "fundamental_constants.H"
#include "physics_numVars.hpp"
#include <AMReX.H>

using Real = amrex::Real;

// enum for unit system, one of CGS, CONSTANTS, CUSTOM
enum class UnitSystem { CGS, CONSTANTS, CUSTOM };

// this struct is specialized by the user application code.
template <typename problem_t> struct Physics_Traits {
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 0;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	// default values for cosmology (can be overridden in specializations)
	static constexpr bool is_cosmology_enabled = false;
	static constexpr double omega_m = 0.315;
	static constexpr double omega_r = 9.2618e-5;
	static constexpr double omega_lambda = 0.685;
	static constexpr double hubble_constant = 1.0;
	static constexpr double a_init = 1.0;
	static constexpr double cosmology_dt_limit = 0.01;
};

// detect if traits exist and provide a default if not (handles missing members in specializations)
template <typename problem_t> struct PhysicsTraits {
	using T = Physics_Traits<problem_t>;

	// Hydro
	static constexpr bool is_hydro_enabled = []() constexpr {
		if constexpr (requires { T::is_hydro_enabled; }) { return T::is_hydro_enabled; }
		return false;
	}();
	static constexpr int numMassScalars = []() constexpr {
		if constexpr (requires { T::numMassScalars; }) { return T::numMassScalars; }
		return 0;
	}();
	static constexpr int numPassiveScalars = []() constexpr {
		if constexpr (requires { T::numPassiveScalars; }) { return T::numPassiveScalars; }
		return numMassScalars;
	}();

	// Radiation / Dust
	static constexpr bool is_radiation_enabled = []() constexpr {
		if constexpr (requires { T::is_radiation_enabled; }) { return T::is_radiation_enabled; }
		return false;
	}();
	static constexpr int nGroups = []() constexpr {
		if constexpr (requires { T::nGroups; }) { return T::nGroups; }
		return 1;
	}();
	static constexpr bool is_dust_enabled = []() constexpr {
		if constexpr (requires { T::is_dust_enabled; }) { return T::is_dust_enabled; }
		return false;
	}();
	static constexpr int nDustGroups = []() constexpr {
		if constexpr (requires { T::nDustGroups; }) { return T::nDustGroups; }
		return 1;
	}();

	// Gravity / MHD
	static constexpr bool is_self_gravity_enabled = []() constexpr {
		if constexpr (requires { T::is_self_gravity_enabled; }) { return T::is_self_gravity_enabled; }
		return false;
	}();
	static constexpr bool is_mhd_enabled = []() constexpr {
		if constexpr (requires { T::is_mhd_enabled; }) { return T::is_mhd_enabled; }
		return false;
	}();

	// Units
	static constexpr UnitSystem unit_system = []() constexpr {
		if constexpr (requires { T::unit_system; }) { return T::unit_system; }
		return UnitSystem::CGS;
	}();
	static constexpr double boltzmann_constant = []() constexpr {
		if constexpr (requires { T::boltzmann_constant; }) { return T::boltzmann_constant; }
		return C::k_B;
	}();
	static constexpr double gravitational_constant = []() constexpr {
		if constexpr (requires { T::gravitational_constant; }) { return T::gravitational_constant; }
		return C::Gconst;
	}();
	static constexpr double c_light = []() constexpr {
		if constexpr (requires { T::c_light; }) { return T::c_light; }
		return C::c_light;
	}();
	static constexpr double radiation_constant = []() constexpr {
		if constexpr (requires { T::radiation_constant; }) { return T::radiation_constant; }
		return C::a_rad;
	}();
	static constexpr double unit_length = []() constexpr {
		if constexpr (requires { T::unit_length; }) { return T::unit_length; }
		return 1.0;
	}();
	static constexpr double unit_mass = []() constexpr {
		if constexpr (requires { T::unit_mass; }) { return T::unit_mass; }
		return 1.0;
	}();
	static constexpr double unit_time = []() constexpr {
		if constexpr (requires { T::unit_time; }) { return T::unit_time; }
		return 1.0;
	}();
	static constexpr double unit_temperature = []() constexpr {
		if constexpr (requires { T::unit_temperature; }) { return T::unit_temperature; }
		return 1.0;
	}();

	// Cosmology
	static constexpr bool is_cosmology_enabled = []() constexpr {
		if constexpr (requires { T::is_cosmology_enabled; }) { return T::is_cosmology_enabled; }
		return false;
	}();
	static constexpr double omega_m = []() constexpr {
		if constexpr (requires { T::omega_m; }) { return T::omega_m; }
		return 0.315;
	}();
	static constexpr double omega_r = []() constexpr {
		if constexpr (requires { T::omega_r; }) { return T::omega_r; }
		return 9.2618e-5;
	}();
	static constexpr double omega_lambda = []() constexpr {
		if constexpr (requires { T::omega_lambda; }) { return T::omega_lambda; }
		return 0.685;
	}();
	static constexpr double hubble_constant = []() constexpr {
		if constexpr (requires { T::hubble_constant; }) { return T::hubble_constant; }
		return 1.0;
	}();
	static constexpr double a_init = []() constexpr {
		if constexpr (requires { T::a_init; }) { return T::a_init; }
		return 1.0;
	}();
	static constexpr double cosmology_dt_limit = []() constexpr {
		if constexpr (requires { T::cosmology_dt_limit; }) { return T::cosmology_dt_limit; }
		return 0.01;
	}();
};




// this struct stores the indices at which quantities start
template <typename problem_t> struct Physics_Indices {
	// number of cc quantities required for advection problems
	static constexpr int nvarTotal_cc_adv = 1;
	// number of cc quantities required for rad /+ hydro problem
	static constexpr int nvarTotal_cc = []() constexpr {
		if constexpr (!(PhysicsTraits<problem_t>::is_hydro_enabled || PhysicsTraits<problem_t>::is_radiation_enabled)) {
			return nvarTotal_cc_adv;
		}
		return PhysicsTraits<problem_t>::numPassiveScalars + Physics_NumVars::numHydroVars +
		       Physics_NumVars::numDustVarsPerGroup * PhysicsTraits<problem_t>::nDustGroups *
			   static_cast<int>(PhysicsTraits<problem_t>::is_dust_enabled) +
		       Physics_NumVars::numRadVarsPerGroup * PhysicsTraits<problem_t>::nGroups *
			   static_cast<int>(PhysicsTraits<problem_t>::is_radiation_enabled);
	}();
	// cell-centered
	static constexpr int hydroFirstIndex = 0;
	static constexpr int pscalarFirstIndex = Physics_NumVars::numHydroVars;
	static constexpr int dustFirstIndex = pscalarFirstIndex + PhysicsTraits<problem_t>::numPassiveScalars;
	static constexpr int radFirstIndex = dustFirstIndex + Physics_NumVars::numDustVarsPerGroup * PhysicsTraits<problem_t>::nDustGroups *
								  static_cast<int>(PhysicsTraits<problem_t>::is_dust_enabled);
	// face-centered
	static constexpr int nvarPerDim_fc = Physics_NumVars::numMHDVars_per_dim * static_cast<int>(PhysicsTraits<problem_t>::is_mhd_enabled);
	static constexpr int nvarTotal_fc = AMREX_SPACEDIM * nvarPerDim_fc;
	static constexpr int mhdFirstIndex = 0;
};

#endif // PHYSICS_INFO_HPP_
