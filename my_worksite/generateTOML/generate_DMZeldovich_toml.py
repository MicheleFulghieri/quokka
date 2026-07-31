#!/usr/bin/env python3
"""
Generate inputs/DMZeldovich.in.toml from inputs/DMZeldovich.in by computing
`comoving_mean_density = mass_reduction_fraction * (3 H0^2 / (8 pi G))` using
constants from extern/Microphysics/constants/fundamental_constants.H.

The generated TOML preserves the full original `DMZeldovich.in` as a commented
block at the top and then provides a TOML mapping with selected keys (cosmology and
universe) including the computed `comoving_mean_density` with full precision.

This script does NOT modify `inputs/DMZeldovich.in`.
"""

import re
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / 'inputs' / 'DMZeldovich.in'
TOML_PATH = ROOT / 'inputs' / 'DMZeldovich.in.toml'
CONSTANTS_PATH = ROOT / 'extern' / 'Microphysics' / 'constants' / 'fundamental_constants.H'

# helpers

def extract_constant(name: str, header: Path):
    pat = re.compile(rf"constexpr\s+amrex::Real\s+{re.escape(name)}\s*=\s*([0-9Ee\+\-\.]+)")
    text = header.read_text()
    m = pat.search(text)
    if not m:
        raise RuntimeError(f"Could not find constant '{name}' in {header}")
    return float(m.group(1))


def find_key_in_file_text(text, key):
    # match lines like 'key = value'
    key_re = re.compile(rf"^{re.escape(key)}\s*=\s*(.*)$", re.MULTILINE)
    m = key_re.search(text)
    if not m:
        return None
    val = m.group(1).split('#')[0].strip()
    try:
        return float(val)
    except Exception:
        return None


def main():
    if not TEMPLATE_PATH.exists():
        raise SystemExit(f"Template {TEMPLATE_PATH} not found")
    tmpl_text = TEMPLATE_PATH.read_text()

    # read values from template
    h = find_key_in_file_text(tmpl_text, 'cosmology.hubble_constant')
    mass_frac = find_key_in_file_text(tmpl_text, 'universe.mass_reduction_fraction')

    if h is None or mass_frac is None:
        raise SystemExit('Could not parse cosmology.hubble_constant or universe.mass_reduction_fraction from template')

    # extract constants
    parsec = extract_constant('parsec', CONSTANTS_PATH)
    Gconst = extract_constant('Gconst', CONSTANTS_PATH)

    Mpc_to_cm = parsec * 1.0e6
    H0_cgs = (h * 100.0 * 1e5) / Mpc_to_cm
    rho_crit_0 = 3.0 * H0_cgs * H0_cgs / (8.0 * math.pi * Gconst)
    comoving_mean = mass_frac * rho_crit_0

    # prepare toml content: first include original as commented block
    commented = '\n'.join('# ' + line for line in tmpl_text.splitlines())

    toml_lines = []
    toml_lines.append('# Generated from inputs/DMZeldovich.in')
    toml_lines.append('# Original file preserved below:')
    toml_lines.append(commented)
    toml_lines.append('')
    toml_lines.append('# Computed values (using constants from: ' + str(CONSTANTS_PATH) + ')')
    toml_lines.append('[cosmology]')
    toml_lines.append(f'hubble_constant = {h!r}')
    toml_lines.append('')
    toml_lines.append('[universe]')
    toml_lines.append(f'mass_reduction_fraction = {mass_frac!r}')
    toml_lines.append(f'comoving_mean_density = {comoving_mean:.17e}')
    toml_lines.append('')
    toml_lines.append('# Notes: comoving_mean_density was computed as:')
    toml_lines.append('# comoving_mean_density = mass_reduction_fraction * (3 * H0^2 / (8 * pi * G))')
    toml_lines.append('# where H0 = (h * 100 km/s/Mpc) converted to s^-1 using parsec from fundamental_constants.H')

    TOML_PATH.write_text('\n'.join(toml_lines) + '\n')

    print('Wrote TOML to', TOML_PATH)
    print('h =', h)
    print('mass_reduction_fraction =', mass_frac)
    print('comoving_mean_density =', f"{comoving_mean:.17e}")
    print('used constants parsec =', parsec, 'Gconst =', Gconst)

if __name__ == '__main__':
    main()
