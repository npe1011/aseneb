import os
import shutil
import copy
import subprocess as sp
from pathlib import Path
import tempfile
from typing import Tuple, List, Optional, Union

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.atoms import Atoms
import ase.io
from ase.units import Hartree, Bohr

from aseneb.utils import popen_bg

# Some internal constants
FLOAT = float
ALL_CHANGES = tuple(all_changes)
XTB_INIT_XYZ_FILE = 'init.xyz'
XTB_ENERGY_FILE = 'energy'
XTB_GRADIENT_FILE = 'gradient'
XTB_CHARGE_FILE = 'charges'
XTB_LOG_FILE = 'xtblog_debug.txt'


class XTBRunTimeError(Exception):
    pass


class XTBParams:
    def __init__(self,
                 method: str = 'gfn2',
                 charge: int = 0,
                 uhf: int = 0,
                 solvation: Optional[str] = None,
                 solvent: Optional[str] = None):

        # check parameters
        try:
            method = method.lower()
            assert method in ['gfn1', 'gfn2', 'gfn0', 'gfnff']
            assert uhf >= 0
            if solvation is not None:
                solvation = solvation.lower()
                assert solvation in ['alpb', 'gbsa']
                assert solvent is not None
                assert len(solvent) > 0
        except AssertionError as e:
            raise ValueError('Given XTB parameters are not valid. ' + str(e.args))

        self.method = method
        self.charge = charge
        self.uhf = uhf
        self.solvation = solvation
        self.solvent = solvent

    @property
    def args(self) -> List[str]:
        _args = ['--' + self.method, '--chrg', str(self.charge), '--uhf', str(self.uhf)]
        if self.solvation is not None:
            _args.extend(['--' + self.solvation, self.solvent])
        return _args


class XTBCalculator(Calculator):
    """
    ASE Calculator depending on external XTB binary without python API, mainly for Windows.
    """
    implemented_properties = [
        'energy',
        'forces',
        'charges',
        'dipole'
    ]

    def __init__(self,
                 xtb_params: XTBParams,
                 workdir: Union[str, Path, None],
                 xtb_bin:  Union[str, Path],
                 xtb_param_dir: Union[str, Path,],
                 omp_num_threads: int,
                 omp_stacksize: str,
                 **kwargs):

        super().__init__(**kwargs)

        self.params: XTBParams = copy.copy(xtb_params)

        self.workdir: Optional[Path] = None
        if workdir is not None:
            self.workdir = Path(workdir).absolute()

        self.xtb_bin: Path = Path(xtb_bin).absolute()
        self.xtb_param_dir = Path(xtb_param_dir).absolute()
        self.omp_num_threads: int = omp_num_threads
        self.omp_stacksize: str = omp_stacksize

    def _set_environ(self) -> None:

        current_path = os.environ.get('PATH', '')
        xtb_bin_dir = str(self.xtb_bin.parent)
        if xtb_bin_dir not in current_path.split(os.pathsep):
            os.environ['PATH'] = current_path + os.pathsep + xtb_bin_dir

        os.environ['XTBPATH'] = str(self.xtb_param_dir)
        omp_num_threads = str(self.omp_num_threads)
        os.environ['OMP_NUM_THREADS'] = omp_num_threads + ',1'
        os.environ['MKL_NUM_THREADS'] = omp_num_threads
        os.environ['OMP_MAX_ACTIVE_LEVELS'] = '1'
        os.environ['OMP_STACKSIZE'] = self.omp_stacksize.rstrip('b').rstrip('B').rstrip('w').rstrip('b')

    def calculate(
            self,
            atoms: Optional[Atoms] = None,
            properties: Tuple[str] = ('energy', 'forces'),
            system_changes: Tuple[str] = ALL_CHANGES,
    ) -> None:

        super().calculate(atoms, properties, system_changes)
        self._set_environ()

        # Run in temporary working directory
        success = False
        prev_dir = Path(os.getcwd())
        workdir = Path(tempfile.mkdtemp(dir=self.workdir))
        try:
            os.chdir(workdir)
            ase.io.write(XTB_INIT_XYZ_FILE, atoms, format='xyz')
            command = [str(self.xtb_bin), XTB_INIT_XYZ_FILE] + self.params.args + ['--grad']
            if self.params.method.lower() not in ['gfnff']:
                command += ['--dipole']
            with open(XTB_LOG_FILE, 'w', encoding='utf-8') as f:
                proc = popen_bg(command, universal_newlines=True, encoding='utf-8', stdout=f, stderr=sp.STDOUT)
                proc.wait()
            with open(XTB_LOG_FILE, 'r', encoding='utf-8') as f:
                log_data = f.read()
                if 'normal termination of xtb' not in log_data:
                    raise XTBRunTimeError('xtb failed in {:}'.format(workdir))
                else:
                    log_data = log_data.splitlines()

            # Get calculated values and convert to ase format.
            self.results['energy'] = _read_xtb_energy(XTB_ENERGY_FILE) * Hartree
            self.results['forces'] = -1.0 * _read_xtb_gradient(XTB_GRADIENT_FILE, len(atoms)) * Hartree / Bohr
            if self.params.method.lower() not in ['gfnff']:
                self.results['charges'] = _read_xtb_charges(XTB_CHARGE_FILE)
                self.results['dipole'] = _read_xtb_dipole(log_data) * Bohr

        except:
            raise

        else:
            success = True

        finally:
            os.chdir(prev_dir)
            if success:
                shutil.rmtree(workdir, ignore_errors=True)


def _read_xtb_energy(energy_file: Union[str, os.PathLike]) -> float:
    energy_file = Path(energy_file)
    with energy_file.open(mode='r') as f:
        energy_data = f.readlines()
    assert energy_data[0].strip().startswith('$energy')
    return float(energy_data[1].strip().split()[1])


def _read_xtb_gradient(gradient_file: Union[str, os.PathLike], num_atom: int) -> np.ndarray:
    gradient_file = Path(gradient_file)
    with gradient_file.open(mode='r') as f:
        gradient_data = f.readlines()
    assert gradient_data[0].strip().startswith('$grad')
    gradient = []
    for line in gradient_data[2+num_atom:2+2*num_atom]:
        gradient.append(line.strip().split())

    return np.array(gradient, dtype=FLOAT)


def _read_xtb_charges(charges_file: Union[str, os.PathLike]) -> np.ndarray:
    charges_file = Path(charges_file)
    with charges_file.open(mode='r') as f:
        charge_data = f.readlines()
    charges = []
    for line in charge_data:
        if line.strip() == '':
            break
        charges.append(float(line.strip()))

    return np.array(charges, dtype=FLOAT)


def _read_xtb_dipole(xtb_log_data: List[str]) -> np.ndarray:
    dipole_line = None
    for i, line in enumerate(xtb_log_data):
        if line.strip().startswith('molecular dipole:'):
            dipole_line = xtb_log_data[i+3].strip()
            break
    if dipole_line is None:
        raise XTBRunTimeError('Reading dipole moment failed.')
    try:
        assert dipole_line.startswith('full:')
        terms = dipole_line.split()
        assert len(terms) == 5
        dipole = np.array([float(terms[1]), float(terms[2]), float(terms[3])], dtype=FLOAT)
    except AssertionError:
        raise XTBRunTimeError('Printed dipole moment format is not expected.')
    else:
        return dipole
