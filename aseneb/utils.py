from typing import List, Union, Any
import os
from pathlib import Path
import subprocess as sp

import ase


def atom_number_to_symbol(atom_number):
    ATOM_LIST = ["bq",
                 "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
                 "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
                 "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb",
                 "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
                 "Er", "Tm", "Yb","Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po",
                 "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
                 "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn"]
    return ATOM_LIST[atom_number]


def atoms_to_text(atoms: ase.atoms.Atoms) -> List[str]:
    """
    convert Atoms object to text in xyz format without header lines, for file.writelines
    """
    text = []
    atomic_numbers = atoms.get_atomic_numbers()
    cartesians = atoms.get_positions()
    for i in range(len(atoms)):
        symbol = atom_number_to_symbol(atomic_numbers[i])
        line = '{:<2s}  {:>12.8f}  {:>12.8f}  {:>12.8f}\n'.format(symbol,
                                                                  cartesians[i,0], cartesians[i,1], cartesians[i,2])
        text.append(line)
    return text


def energy_conversion(energies: Any, energy_unit: str) -> Any:
    """
    convert energies (in eV) to given unit.
    energy_unit should be kcal/mol, kJ/mol, Hartree, or eV (str, case-insensitive)
    """
    conv = 1.00
    if energy_unit.lower() == 'kcal/mol':
        conv = 23.06031
    elif energy_unit.lower() == 'kj/mol':
        conv = 96.48534
    elif energy_unit.lower() in ['hartrees', 'hartree', 'au', 'a.u.']:
        conv = 3.674932e-2
    elif energy_unit.lower() == 'eV':
        pass
    else:
        raise RuntimeError('energy_unit should be kcal/mol, eV, or kJ/mol')
    return energies * conv


def popen_bg(*args, **kwargs):
    if os.name == 'nt':
        startupinfo = sp.STARTUPINFO()
        startupinfo.dwFlags |= sp.STARTF_USESHOWWINDOW
        win_kwargs = {'startupinfo': startupinfo}
        return sp.Popen(*args, **kwargs, **win_kwargs)
    else:
        return sp.Popen(*args, **kwargs)


def remove(file: Union[Path, str], ignore_errors=False):
    if os.path.exists(file):
        try:
            os.remove(file)
        except:
            if ignore_errors:
                pass
            else:
                raise
