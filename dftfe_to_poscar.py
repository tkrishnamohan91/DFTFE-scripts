#!/usr/bin/env python3
"""
DFT-FE → POSCAR converter (coord.inp format)
--------------------------------------------
Inputs:
  - domainVectors.inp : 3 lattice vectors in Bohr (a0) → converted to Å
  - coord.inp         : lines with
                        Zval  valence_electrons  x  y  z  [magnetic_moment_optional]
                        where x y z are fractional if --coords-type direct,
                        or Cartesian if --coords-type cartesian

Ignored for POSCAR:
  - valence_electrons column
  - magnetic_moment (optional 6th column)

Defaults:
  - --coords-type direct
  - --poscar-type direct
  - lattice units Bohr (converted to Å)

Examples
--------
Default (Direct → Direct):
  python dftfe_to_poscar.py -l domainVectors.inp -c coord.inp -o POSCAR

Cartesian coords in Bohr:
  python dftfe_to_poscar.py -l domainVectors.inp -c coord.inp --coords-type cartesian -o POSCAR

Cartesian coords already in Å:
  python dftfe_to_poscar.py -l domainVectors.inp -c coord.inp --coords-type cartesian --units angstrom -o POSCAR

Force species order + wrap fractional into [0,1):
  python dftfe_to_poscar.py -l domainVectors.inp -c coord.inp --species-order La Zr Li O --wrap -o POSCAR
"""

from __future__ import annotations
import argparse
import re
import sys
import math
from typing import List, Tuple, Dict, Iterable

BOHR_TO_ANG = 0.529177210903

# Periodic table (1..94); extend if you need >U
PTABLE = {
    '1':'H','2':'He','3':'Li','4':'Be','5':'B','6':'C','7':'N','8':'O','9':'F','10':'Ne',
    '11':'Na','12':'Mg','13':'Al','14':'Si','15':'P','16':'S','17':'Cl','18':'Ar','19':'K','20':'Ca',
    '21':'Sc','22':'Ti','23':'V','24':'Cr','25':'Mn','26':'Fe','27':'Co','28':'Ni','29':'Cu','30':'Zn',
    '31':'Ga','32':'Ge','33':'As','34':'Se','35':'Br','36':'Kr','37':'Rb','38':'Sr','39':'Y','40':'Zr',
    '41':'Nb','42':'Mo','43':'Tc','44':'Ru','45':'Rh','46':'Pd','47':'Ag','48':'Cd','49':'In','50':'Sn',
    '51':'Sb','52':'Te','53':'I','54':'Xe','55':'Cs','56':'Ba','57':'La','58':'Ce','59':'Pr','60':'Nd',
    '61':'Pm','62':'Sm','63':'Eu','64':'Gd','65':'Tb','66':'Dy','67':'Ho','68':'Er','69':'Tm','70':'Yb',
    '71':'Lu','72':'Hf','73':'Ta','74':'W','75':'Re','76':'Os','77':'Ir','78':'Pt','79':'Au','80':'Hg',
    '81':'Tl','82':'Pb','83':'Bi','84':'Po','85':'At','86':'Rn','87':'Fr','88':'Ra','89':'Ac','90':'Th',
    '91':'Pa','92':'U','93':'Np','94':'Pu'
}

COMMENT_PREFIXES = ('#','!',';')

def _is_int(s: str) -> bool:
    return re.fullmatch(r"\d+", s) is not None

def _to_symbol_from_Zval(zstr: str) -> str:
    if not _is_int(zstr):
        raise ValueError(f"[coords] Zval must be integer atomic number, got '{zstr}'")
    sym = PTABLE.get(zstr)
    if not sym:
        raise ValueError(f"[coords] Zval {zstr} not in PTABLE (extend table if needed).")
    return sym

def read_lattice(path: str, lattice_units: str = 'bohr') -> List[List[float]]:
    """Read 3 lattice vectors; convert to Å if input is Bohr."""
    vecs: List[List[float]] = []
    with open(path, 'r') as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith(COMMENT_PREFIXES):
                continue
            parts = re.split(r"[\s,;]+", s)
            nums = [float(x) for x in parts if x]
            if len(nums) >= 3:
                vecs.append(nums[:3])
            if len(vecs) == 3:
                break
    if len(vecs) != 3:
        raise ValueError(f"[lattice] expected 3 vectors in '{path}', got {len(vecs)}")
    if lattice_units.lower() in ('bohr','a0','au','atomic'):
        for v in vecs:
            for i in range(3):
                v[i] *= BOHR_TO_ANG
    return vecs  # in Å

def read_coordinates_coord_inp(path: str,
                               coords_type: str = 'direct',
                               units: str = 'bohr') -> List[Tuple[str, List[float]]]:
    """
    Parse coord.inp format:
      Zval  valence_electrons  x  y  z  [magmom]
    Returns list of (symbol, [x, y, z]).
    """
    atoms: List[Tuple[str, List[float]]] = []
    with open(path, 'r') as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith(COMMENT_PREFIXES):
                continue
            parts = re.split(r"[\s,;]+", s)
            if len(parts) < 5:
                # allow an optional single atom-count header line; skip it
                if len(parts) == 1 and _is_int(parts[0]) and not atoms:
                    continue
                # otherwise ignore
                continue
            zval = parts[0]
            # parts[1] = valence electrons (ignored)
            try:
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                continue
            sym = _to_symbol_from_Zval(zval)
            atoms.append((sym, [x, y, z]))

    if not atoms:
        raise ValueError(f"[coords] no atoms parsed from '{path}'")

    # If Cartesian in Bohr, convert positions to Å
    if coords_type.lower() == 'cartesian' and units.lower() in ('bohr','a0','au','atomic'):
        for _, r in atoms:
            r[0] *= BOHR_TO_ANG
            r[1] *= BOHR_TO_ANG
            r[2] *= BOHR_TO_ANG
    return atoms

def det3(m):
    return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
          - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
          + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))

def inv3(m):
    D = det3(m)
    if abs(D) < 1e-14:
        raise ValueError("Singular lattice matrix; cannot invert.")
    return [
        [(m[1][1]*m[2][2]-m[1][2]*m[2][1])/D, -(m[0][1]*m[2][2]-m[0][2]*m[2][1])/D,  (m[0][1]*m[1][2]-m[0][2]*m[1][1])/D],
        [-(m[1][0]*m[2][2]-m[1][2]*m[2][0])/D,  (m[0][0]*m[2][2]-m[0][2]*m[2][0])/D, -(m[0][0]*m[1][2]-m[0][2]*m[1][0])/D],
        [(m[1][0]*m[2][1]-m[1][1]*m[2][0])/D,  -(m[0][0]*m[2][1]-m[0][1]*m[2][0])/D,  (m[0][0]*m[1][1]-m[0][1]*m[1][0])/D],
    ]

def matmul33(m, v):
    return [m[0][0]*v[0]+m[0][1]*v[1]+m[0][2]*v[2],
            m[1][0]*v[0]+m[1][1]*v[1]+m[1][2]*v[2],
            m[2][0]*v[0]+m[2][1]*v[1]+m[2][2]*v[2]]

def cart_to_frac(L_rows, r_cart):
    # r_cart = A * r_frac  ⇒ r_frac = A^{-1} * r_cart, where columns of A are lattice vectors
    A = [[L_rows[0][0], L_rows[1][0], L_rows[2][0]],
         [L_rows[0][1], L_rows[1][1], L_rows[2][1]],
         [L_rows[0][2], L_rows[1][2], L_rows[2][2]]]
    return matmul33(inv3(A), r_cart)

def frac_to_cart(L_rows, r_frac):
    A = [[L_rows[0][0], L_rows[1][0], L_rows[2][0]],
         [L_rows[0][1], L_rows[1][1], L_rows[2][1]],
         [L_rows[0][2], L_rows[1][2], L_rows[2][2]]]
    return matmul33(A, r_frac)

def wrap01(vals: Iterable[float]) -> List[float]:
    return [v - math.floor(v) for v in vals]  # wrap into [0,1)

def order_species(atoms: List[Tuple[str, List[float]]], user_order: List[str] | None) -> List[str]:
    if user_order:
        # Normalize symbols (accept either symbol or Zval)
        norm = []
        for s in user_order:
            if _is_int(s):
                norm.append(_to_symbol_from_Zval(s))
            else:
                norm.append(s[0].upper()+s[1:].lower())
        return norm
    seen: Dict[str, bool] = {}
    order: List[str] = []
    for s, _ in atoms:
        if s not in seen:
            seen[s] = True
            order.append(s)
    return order

def group_atoms(atoms: List[Tuple[str, List[float]]], order: List[str]) -> Dict[str, List[List[float]]]:
    groups: Dict[str, List[List[float]]] = {s: [] for s in order}
    for s, r in atoms:
        if s in groups:
            groups[s].append(r)
        else:
            raise ValueError(f"[coords] species '{s}' not in header order {order}. "
                             "Provide --species-order to include it.")
    return groups

def write_poscar(out_path: str,
                 title: str,
                 lattice: List[List[float]],
                 species_order: List[str],
                 groups: Dict[str, List[List[float]]],
                 poscar_type: str,
                 coords_type_in: str,
                 do_wrap: bool) -> None:
    with open(out_path, 'w') as f:
        f.write(f"{title}\n")
        f.write("1.0\n")
        for v in lattice:
            f.write(f"{v[0]:20.16f} {v[1]:20.16f} {v[2]:20.16f}\n")
        f.write(" ".join(species_order) + "\n")
        f.write(" ".join(str(len(groups[s])) for s in species_order) + "\n")
        f.write(("Direct" if poscar_type == 'direct' else 'Cartesian') + "\n")
        for s in species_order:
            for r in groups[s]:
                out_vec = r
                if poscar_type == 'direct' and coords_type_in == 'cartesian':
                    out_vec = cart_to_frac(lattice, r)
                elif poscar_type == 'cartesian' and coords_type_in == 'direct':
                    out_vec = frac_to_cart(lattice, r)
                if poscar_type == 'direct' and do_wrap:
                    out_vec = wrap01(out_vec)
                f.write(f"{out_vec[0]:.16f} {out_vec[1]:.16f} {out_vec[2]:.16f}\n")

def main():
    ap = argparse.ArgumentParser(description="Convert DFT-FE domainVectors.inp + coord.inp to VASP POSCAR.")
    ap.add_argument('-l','--lattice', required=True, help="domainVectors.inp (Bohr by default)")
    ap.add_argument('-c','--coords',  required=True, help="coord.inp (Zval  val_e  x  y  z  [magmom])")
    ap.add_argument('-o','--outfile', default='POSCAR', help="Output POSCAR filepath")
    ap.add_argument('--title', default='Generated by dftfe_to_poscar.py', help='POSCAR title/comment line')

    ap.add_argument('--coords-type', choices=['cartesian','direct'], default='direct',
                    help="Type of positions in coord.inp (default: direct)")
    ap.add_argument('--units', choices=['angstrom','bohr'], default='bohr',
                    help="Units for *Cartesian* coordinates (ignored for Direct); default: bohr")
    ap.add_argument('--poscar-type', choices=['cartesian','direct'], default='direct',
                    help="Coordinate type to write in POSCAR (default: direct)")
    ap.add_argument('--species-order', nargs='+',
                    help="Force species order in POSCAR header (accepts symbols or Zvals), e.g. La Zr Li O")
    ap.add_argument('--wrap', action='store_true',
                    help="Wrap fractional coordinates into [0,1) when writing Direct")

    a = ap.parse_args()

    try:
        lattice = read_lattice(a.lattice, lattice_units='bohr')
        atoms   = read_coordinates_coord_inp(a.coords, coords_type=a.coords_type, units=a.units)
        order   = order_species(atoms, a.species_order)
        groups  = group_atoms(atoms, order)

        write_poscar(out_path=a.outfile,
                     title=a.title,
                     lattice=lattice,
                     species_order=order,
                     groups=groups,
                     poscar_type=a.poscar_type,
                     coords_type_in=a.coords_type,
                     do_wrap=a.wrap)

        print(f"Wrote POSCAR ({a.poscar_type}; species: {' '.join(order)})")
    except Exception as e:
        sys.exit(f"Error: {e}")

if __name__ == "__main__":
    main()

