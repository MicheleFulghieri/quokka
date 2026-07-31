#!/usr/bin/env python3
"""
Generate DMZeldovich.in from a TOML override by computing comoving_mean_density
as mass_reduction_fraction * (3 H0^2 / (8 pi G)) using the constants in
extern/Microphysics/constants/fundamental_constants.H to ensure exact matching.

The script preserves all comments and values in the original inputs/DMZeldovich.in,
creates a backup (inputs/DMZeldovich.in.bak) and replaces the line starting with
`comoving_mean_density` with the computed numeric value (keeping any trailing comment).

Usage:
  python3 scripts/generate_DMZeldovich_input.py

"""

import re
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOML_PATH = ROOT / 'inputs' / 'DMZeldovich.in.toml'
TEMPLATE_PATH = ROOT / 'inputs' / 'DMZeldovich.in'
BACKUP_PATH = ROOT / 'inputs' / 'DMZeldovich.in.bak'
CONSTANTS_PATH = ROOT / 'extern' / 'Microphysics' / 'constants' / 'fundamental_constants.H'

# Minimal TOML-like parser for the keys we need (no external deps)

def parse_simple_toml(p: Path):
    data = {}
    if not p.exists():
        return data
    key_re = re.compile(r"^([A-Za-z0-9_\.]+)\s*=\s*(.*)$")
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            m = key_re.match(line)
            if not m:
                continue
            key, val = m.group(1), m.group(2)
            # strip quotes
            val = val.split('#')[0].strip()
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            try:
                if '.' in val or 'e' in val or 'E' in val:
                    v = float(val)
                else:
                    v = int(val)
            except Exception:
                v = val
            data[key] = v
    return data


def extract_constant(name: str, header: Path):
    # look for lines like: constexpr amrex::Real parsec = 3.085677587679311e18;  // cm
    pat = re.compile(rf"constexpr\s+amrex::Real\s+{re.escape(name)}\s*=\s*([0-9Ee\+\-\.]+)")
    text = header.read_text()
    m = pat.search(text)
    if not m:
        raise RuntimeError(f"Could not find constant '{name}' in {header}")
    return float(m.group(1))


def find_key_in_template(lines, key):
    # return index and parts (pre, sep, value, comment)
    # match lines that start with key (possibly with spaces) and an '='
    key_re = re.compile(rf"^(\s*)({re.escape(key)})\s*(=)\s*(.*)$")
    for idx, line in enumerate(lines):
        m = key_re.match(line)
        if m:
            pre = m.group(1)
            k = m.group(2)
            sep = m.group(3)
            rest = m.group(4)
            # split rest into value and trailing comment
            if '#' in rest:
                val, comment = rest.split('#', 1)
                comment = '#' + comment
            else:
                val = rest
                comment = ''
            return idx, pre, k, sep, val.strip(), comment
    return None


def main():
    toml = parse_simple_toml(TOML_PATH)

    # read template
    if not TEMPLATE_PATH.exists():
        raise SystemExit(f"Template {TEMPLATE_PATH} not found")
    text = TEMPLATE_PATH.read_text()
    lines = text.splitlines()

    # Try to get h and mass_reduction_fraction from TOML first, else from template
    h = None
    mass_frac = None

    if 'cosmology.hubble_constant' in toml:
        h = float(toml['cosmology.hubble_constant'])
    if 'universe.mass_reduction_fraction' in toml:
        mass_frac = float(toml['universe.mass_reduction_fraction'])

    # parse values from template if needed
    if h is None:
        res = find_key_in_template(lines, 'cosmology.hubble_constant')
        if res:
            _, _, _, _, val, _ = res
            try:
                h = float(val)
            except Exception:
                pass

    if mass_frac is None:
        res = find_key_in_template(lines, 'universe.mass_reduction_fraction')
        if res:
            _, _, _, _, val, _ = res
            try:
                mass_frac = float(val)
            except Exception:
                pass

    if h is None or mass_frac is None:
        raise SystemExit('Could not determine both cosmology.hubble_constant and universe.mass_reduction_fraction')

    # extract constants from header to match C++ exactly
    parsec = extract_constant('parsec', CONSTANTS_PATH)
    Gconst = extract_constant('Gconst', CONSTANTS_PATH)

    Mpc_to_cm = parsec * 1.0e6
    H0_cgs = (h * 100.0 * 1e5) / Mpc_to_cm
    rho_crit_0 = 3.0 * H0_cgs * H0_cgs / (8.0 * math.pi * Gconst)
    comoving_mean = mass_frac * rho_crit_0

    # prepare formatted string with full precision
    comoving_str = f"{comoving_mean:.17e}"

    # find comoving_mean_density line in template
    found = find_key_in_template(lines, 'comoving_mean_density')
    if found:
        idx, pre, key, sep, val, comment = found
        new_line = f"{pre}{key} {sep} {comoving_str}"
        if comment:
            new_line += ' ' + comment
        lines[idx] = new_line
    else:
        # insert after universe.mass_reduction_fraction if present
        res = find_key_in_template(lines, 'universe.mass_reduction_fraction')
        insert_at = len(lines)
        if res:
            insert_at = res[0] + 1
        new_line = f"comoving_mean_density = {comoving_str}"
        lines.insert(insert_at, new_line)

    # backup
    BACKUP_PATH.write_text(text)

    # write output
    out_text = '\n'.join(lines) + '\n'
    out_path = TEMPLATE_PATH
    out_path.write_text(out_text)

    print('Wrote', out_path)
    print('Backup saved to', BACKUP_PATH)
    print('Using constants from', CONSTANTS_PATH)
    print('parsec =', parsec)
    print('Gconst =', Gconst)
    print('h =', h)
    print('mass_reduction_fraction =', mass_frac)
    print('Computed comoving_mean_density =', comoving_str)


if __name__ == '__main__':
    main()
