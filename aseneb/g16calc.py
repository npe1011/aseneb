import os
import subprocess as sp
from pathlib import Path
from typing import Tuple, List, Optional, Union

import numpy as np
import ase.io
from ase.calculators.calculator import Calculator, all_changes
from ase.atoms import Atoms

import config
from aseneb.utils import atoms_to_text, popen_bg

# Some internal constants
FLOAT = float
ALL_CHANGES = tuple(all_changes)


class G16Calculator(Calculator):
    """
    ASE Calculator depending on external g16 binary.
    """
    implemented_properties = [
        'energy',
        'forces',
        'dipole'
    ]

    def __init__(self,
                 template_file: Union[Path, str],
                 job_name: str,
                 num_procs: int = 1,
                 memory: str = '1GB',
                 workdir: Optional[Path] = None,
                 **kwargs):

        super().__init__(**kwargs)
        self.template_file = Path(template_file).absolute()
        if not self.template_file.exists():
            raise FileNotFoundError('Gaussian job template file is not found.')
        if workdir is not None:
            self.workdir = workdir.absolute()
            if not workdir.exists():
                workdir.mkdir(parents=True)
        else:
            self.workdir = Path.cwd().absolute()
        self.job_name = job_name
        self.num_procs = num_procs
        self.memory = memory

        if config.G16_SCRATCH_DIR is None:
            self.scratch_dir = self.workdir
        else:
            self.scratch_dir = Path(config.G16_SCRATCH_DIR).absolute()

    def calculate(
            self,
            atoms: Optional[Atoms] = None,
            properties: List[str] = ['energy'],
            system_changes: Tuple[str] = ALL_CHANGES,
    ) -> None:

        super().calculate(atoms, properties, system_changes)

        # Run in working directory
        prev_dir = Path(os.getcwd())
        try:
            os.chdir(self.workdir)

            # check is this a new calculation required? => comparison with the previous file
            prev_log_file = None
            if Path(self.job_name + '.out').exists():
                prev_log_file = Path(self.job_name + '.out')
            elif Path(self.job_name + '.log').exists():
                prev_log_file = Path(self.job_name + '.log')
            if prev_log_file is not None:
                prev_atoms: Atoms = ase.io.read(prev_log_file, format='gaussian-out')
                if _compare_atoms_coordinates(atoms, prev_atoms):
                    # In case the coordinates are same, just read the previous log.
                    self.results['energy'] = prev_atoms.get_potential_energy()
                    self.results['forces'] = prev_atoms.get_forces()
                    self.results['dipole'] = prev_atoms.get_dipole_moment()
                    return

            # When a new calculation is required:
            with open(self.job_name + '.gjf', mode='w', encoding='utf-8', newline='\n') as f:
                f.writelines(self._prepare_gjf_data(atoms))
            # remove previous log to prevent reading wrong file
            if Path(self.job_name + '.log').exists():
                Path(self.job_name + '.log').unlink()
            if Path(self.job_name + '.out').exists():
                Path(self.job_name + '.out').unlink()

            # Clean-up to ensure no previous scratch files remain.
            files = [self.scratch_dir / (self.job_name + '.rwf'),
                     self.scratch_dir / (self.job_name + '.int'),
                     self.scratch_dir / (self.job_name + '.d2e')]
            for file in files:
                try:
                    file.unlink()
                except:
                    pass

            command = ['g16', self.job_name + '.gjf']
            sp.run(command, universal_newlines=True, encoding='utf-8', stdout=sp.DEVNULL, stderr=sp.DEVNULL)

            if Path(self.job_name + '.out').exists():
                log_file = self.job_name + '.out'
            elif Path(self.job_name + '.log').exists():
                log_file = self.job_name + '.log'
            else:
                raise RuntimeError('g16 log file is not found. Something goes wrong.')

            result_atoms: Atoms = ase.io.read(log_file, format='gaussian-out')

            # Get calculated values and convert to ase format.
            self.results['energy'] = result_atoms.get_potential_energy()
            self.results['forces'] = result_atoms.get_forces()
            self.results['dipole'] = result_atoms.get_dipole_moment()

        except:
            raise

        finally:
            os.chdir(prev_dir)
            files = [self.scratch_dir / (self.job_name + '.rwf'),
                     self.scratch_dir / (self.job_name + '.int'),
                     self.scratch_dir / (self.job_name + '.d2e')]
            for file in files:
                try:
                    file.unlink()
                except:
                    pass

    def _prepare_gjf_data(self, atoms: Atoms) -> List[str]:

        data = [
            '%nprocshared={:}\n'.format(self.num_procs),
            '%mem={:}\n'.format(self.memory),
            '%rwf={:}.rwf\n'.format(self.scratch_dir / self.job_name),
            '%int={:}.int\n'.format(self.scratch_dir / self.job_name),
            '%d2e={:}.d2e\n'.format(self.scratch_dir / self.job_name),
            '%nosave\n',
            '%chk={:}.chk\n'.format(self.job_name)
        ]

        previous_chk = Path(self.job_name + '.chk').exists()

        with self.template_file.open() as f:
            for line in f.readlines():
                if line.strip().startswith('%'):
                    continue
                elif line.strip().startswith('#'):
                    if previous_chk:
                        data.append(line.rstrip() + ' nosymm Force guess=read\n')
                    else:
                        data.append(line.rstrip() + ' nosymm Force\n')
                elif line.strip().lower().startswith('@'):
                    data.extend(atoms_to_text(atoms))
                    continue
                else:
                    data.append(line)

        return data

    @staticmethod
    def set_environment(g16_root: Union[Path, str]):
        g16_root = str(g16_root)
        os.environ['GAUSS_EXEDIR'] = g16_root
        current_path = os.environ.get('PATH')
        if g16_root not in current_path.split(os.pathsep):
            os.environ['PATH'] = current_path + os.pathsep + g16_root


def _compare_atoms_coordinates(atoms1: Atoms, atoms2: Atoms) -> bool:
    """
    return True if two atoms objects have the almost same coordinates.
    """
    if not np.all(atoms1.get_atomic_numbers() == atoms2.get_atomic_numbers()):
        return False
    return np.allclose(atoms1.get_positions(), atoms2.get_positions(), atol=1.0e-6)


def run_g16(gjf_file: str):
    command = ['g16', gjf_file]
    sp.run(command, universal_newlines=True, encoding='utf-8', stdout=sp.DEVNULL, stderr=sp.DEVNULL)

def check_g16_termination(log_file: Path):
    with log_file.open(mode='r') as f:
        for line in f.readlines():
            if 'Normal termination of Gaussian' in line:
                return True
    return False
