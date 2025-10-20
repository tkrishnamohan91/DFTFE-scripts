
#!/usr/bin/env python3
"""
POSCAR -> DFT-FE converter

Creates two files:
  - domainBoundingVectors.inp   (3x3 lattice vectors in Bohr)
  - coordinates.inp             (Z  valence  fx  fy  fz  [mag])

Usage:
  python poscar_to_dftfe.py POSCAR [--outdir PATH] [--quiet] [--magnetic|--non-magnetic]
"""
import argparse, sys
from pathlib import Path

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903

PERIODIC_TABLE = {
    'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,
    'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,
    'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,'Zn':30,
    'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Kr':36,'Rb':37,'Sr':38,'Y':39,'Zr':40,
    'Nb':41,'Mo':42,'Tc':43,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,'In':49,'Sn':50,
    'Sb':51,'Te':52,'I':53,'Xe':54,'Cs':55,'Ba':56,'La':57,'Ce':58,'Pr':59,'Nd':60,
    'Pm':61,'Sm':62,'Eu':63,'Gd':64,'Tb':65,'Dy':66,'Ho':67,'Er':68,'Tm':69,'Yb':70,
    'Lu':71,'Hf':72,'Ta':73,'W':74,'Re':75,'Os':76,'Ir':77,'Pt':78,'Au':79,'Hg':80,
    'Tl':81,'Pb':82,'Bi':83,'Po':84,'At':85,'Rn':86,'Fr':87,'Ra':88,'Ac':89,'Th':90,
    'Pa':91,'U':92,'Np':93,'Pu':94,'Am':95,'Cm':96,'Bk':97,'Cf':98,'Es':99,'Fm':100,
    'Md':101,'No':102,'Lr':103,'Rf':104,'Db':105,'Sg':106,'Bh':107,'Hs':108,'Mt':109,'Ds':110,
    'Rg':111,'Cn':112,'Nh':113,'Fl':114,'Mc':115,'Lv':116,'Ts':117,'Og':118
}

def parse_poscar(path: Path):
    lines = [l.rstrip() for l in path.read_text().splitlines() if l.strip() != '']
    comment = lines[0]
    scale = float(lines[1].split()[0])
    a = [float(x) for x in lines[2].split()[:3]]
    b = [float(x) for x in lines[3].split()[:3]]
    c = [float(x) for x in lines[4].split()[:3]]
    tokens6 = lines[5].split()
    has_symbols = not all(t.replace('.', '', 1).replace('-', '', 1).isdigit() for t in tokens6)
    if has_symbols:
        symbols = tokens6
        counts = [int(x) for x in lines[6].split()]
        cursor = 7
    else:
        symbols = []
        counts = [int(x) for x in tokens6]
        cursor = 6
    if lines[cursor].strip().lower().startswith('s'):
        cursor += 1
    coord_type = lines[cursor].strip().lower()[0]; cursor += 1
    natoms = sum(counts)
    positions = [[float(x) for x in lines[cursor+i].split()[:3]] for i in range(natoms)]
    import numpy as np
    lat = np.array([a,b,c],float)
    if scale>0: lat*=scale
    else:
        target_vol=abs(scale); vol=np.dot(np.cross(lat[0],lat[1]),lat[2])
        lat*= (target_vol/vol)**(1/3)
    if coord_type=='c':
        import numpy as np
        invlat=np.linalg.inv(lat.T)
        positions=[(invlat@np.array(p)).tolist() for p in positions]
    return {'lattice_angs':lat,'symbols':symbols,'counts':counts,'positions_frac':positions}

def prompt_elements(symbols, counts, quiet=False):
    if not symbols:
        if quiet: raise ValueError("Symbols missing in POSCAR.")
        print("Enter symbols for",counts)
        symbols=input("Symbols: ").split()
    elem_info={}
    for sym in symbols:
        z_default=PERIODIC_TABLE.get(sym)
        Z=int(input(f"Atomic number for {sym} [{z_default}]: ") or z_default)
        val=int(input(f"Valence electrons for {sym}: "))
        elem_info[sym]={'Z':Z,'valence':val}
    return elem_info,symbols

def prompt_magnetization(symbols):
    default=float(input("Default starting magnetization [0.0]: ") or 0.0)
    mags={}
    for s in symbols:
        m=input(f"Mag for {s} [{default}]: ")
        mags[s]=float(m) if m else default
    return mags

def build_species(symbols,counts):
    res=[]
    for s,n in zip(symbols,counts): res.extend([s]*n)
    return res

def write_domain_vectors(lat_angs,path:Path):
    with path.open('w') as f:
        for v in lat_angs:
            f.write(f"{v[0]*ANGSTROM_TO_BOHR: .8E}   {v[1]*ANGSTROM_TO_BOHR: .8E}   {v[2]*ANGSTROM_TO_BOHR: .8E}\n")

def write_coordinates(species,positions,elem_info,mag_per_elem,path:Path,magnetic=True):
    with path.open('w') as f:
        for s,(fx,fy,fz) in zip(species,positions):
            Z=elem_info[s]['Z']; val=elem_info[s]['valence']
            if magnetic:
                f.write(f"{Z:2d}  {val:3d}   {fx: .16f}  {fy: .16f}  {fz: .16f}  {mag_per_elem.get(s,0.0):.6f}\n")
            else:
                f.write(f"{Z:2d}  {val:3d}   {fx: .16f}  {fy: .16f}  {fz: .16f}\n")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("poscar",type=Path)
    ap.add_argument("--outdir",type=Path,default=Path("."))
    ap.add_argument("--quiet",action="store_true")
    ap.add_argument("--magnetic",action="store_true",help="Force spin-polarized calculation")
    ap.add_argument("--non-magnetic",action="store_true",help="Force non-spin-polarized calculation")
    a=ap.parse_args()
    d=parse_poscar(a.poscar)
    elem_info,symbols=prompt_elements(d['symbols'],d['counts'],quiet=a.quiet)
    if a.magnetic and a.non_magnetic:
        print("Cannot specify both --magnetic and --non-magnetic"); sys.exit(1)
    if a.magnetic:
        mags=prompt_magnetization(symbols)
        magnetic=True
    elif a.non_magnetic:
        mags={s:0.0 for s in symbols}
        magnetic=False
    else:
        ans=input("Magnetic? [y/N]: ").lower()
        if ans.startswith('y'):
            mags=prompt_magnetization(symbols)
            magnetic=True
        else:
            mags={s:0.0 for s in symbols}
            magnetic=False
    species=build_species(symbols,d['counts'])
    a.outdir.mkdir(exist_ok=True,parents=True)
    write_coordinates(species,d['positions_frac'],elem_info,mags,a.outdir/"coordinates.inp",magnetic=magnetic)
    write_domain_vectors(d['lattice_angs'],a.outdir/"domainBoundingVectors.inp")
    print("Wrote coordinates.inp and domainBoundingVectors.inp")

if __name__=="__main__": main()
