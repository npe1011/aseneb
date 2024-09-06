import os
from typing import Union, List, Optional
from pathlib import Path
from multiprocessing import Process, Pool

import ase.io
from ase.atoms import Atoms
from ase.mep import NEB
from ase.optimize import LBFGS, LBFGSLineSearch, BFGSLineSearch, FIRE

from aseneb.neb_project import NEBProject
from aseneb.palneb import PalNEB
from aseneb.g16calc import G16Calculator, run_g16
from aseneb.ase_result import NEBResult, SingleTrajectory
from aseneb.utils import remove
import config


class NEBProjectNonBlocking(NEBProject):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calculation_process: Optional[Process] = None

    def run_neb(self, prev_number: Optional[int] = None) -> bool:
        """
        Run NEB calculation from the prev_number path. If None, the latest one is read and rerun NEB optimization.
        """
        if self.calculation_process is not None:
            return False
        if prev_number is None:
            prev_number = self.current_final_neb_number()
        if prev_number == 0:
            prev_file = self.initial_path_traj_file()
        else:
            prev_file = self.neb_path_traj_file(prev_number)

        if not prev_file.exists():
            raise RuntimeError(str(prev_file) + ' is not found.')

        # output files
        trajectory_file = self.neb_path_traj_file(prev_number + 1)
        log_file = self.neb_path_log_file(prev_number + 1)
        xyz_file = self.neb_path_xyz_file(prev_number + 1)
        optimized_xyz_file = self.neb_path_optimized_xyz_file(prev_number + 1)

        # Clear output files
        remove(trajectory_file)
        remove(log_file)
        remove(xyz_file)
        remove(optimized_xyz_file)

        # read previous file and prepare for NEB
        prev_result = NEBResult(trajectory_file=prev_file, num_nodes=self.num_images+2)
        new_nodes = []
        for i, _atoms in enumerate(prev_result.atoms_list_list[-1]):
            atoms = _atoms.copy()
            calc = self.get_calculator()
            if self.calculator_type == 'g16':
                calc.job_name = 'node{:0>3d}'.format(i)
            atoms.set_calculator(calc)
            new_nodes.append(atoms)

        if self.neb_parallel > 1:
            neb = PalNEB(images=new_nodes,
                         num_processes=self.neb_parallel,
                         k=self.neb_k,
                         climb=self.neb_climb,
                         remove_rotation_and_translation=True,
                         method=self.neb_method)
        else:
            neb = NEB(images=new_nodes,
                      k=self.neb_k,
                      climb=self.neb_climb,
                      remove_rotation_and_translation=True,
                      method=self.neb_method)

        # optimizer settings and run as a subprocess
        neb_params = (
            neb,
            self.neb_optimizer,
            trajectory_file,
            log_file,
            xyz_file,
            optimized_xyz_file,
            self.neb_fmax,
            self.neb_steps,
            self.num_images+2
        )
        self.calculation_process = Process(target=_run_neb, args=neb_params, name='neb'+str(prev_number+1))
        self.calculation_process.start()

        return True

    def load_init_structure(self, file: Union[str, Path]) -> bool:
        if self.calculation_process is not None:
            return False

        atoms = ase.io.read(file, index=-1)
        calc = self.get_calculator()
        if self.calculator_type == 'g16':
            calc.job_name = 'init'
        atoms.set_calculator(calc)
        remove(self.init_traj_file())
        remove(self.init_log_file())
        remove(self.init_xyz_file())
        if self.opt_init:
            opt_params = (
                atoms,
                self.init_traj_file(),
                self.init_log_file(),
                self.init_xyz_file(),
            )
            self.calculation_process = Process(target=_run_opt, args=opt_params, name='opt_init')
            self.calculation_process.start()
        else:
            sp_params = (
                atoms,
                self.init_traj_file(),
                self.init_xyz_file(),
            )
            self.calculation_process = Process(target=_run_single_point, args=sp_params, name='sp_init')
            self.calculation_process.start()

        return True

    def load_final_structure(self, file: Union[str, Path]) -> bool:
        if self.calculation_process is not None:
            return False

        atoms = ase.io.read(file, index=-1)
        calc = self.get_calculator()
        if self.calculator_type == 'g16':
            calc.job_name = 'final'
        atoms.set_calculator(calc)
        remove(self.final_traj_file())
        remove(self.final_log_file())
        remove(self.final_xyz_file())
        if self.opt_final:
            opt_params = (
                atoms,
                self.final_traj_file(),
                self.final_log_file(),
                self.final_xyz_file(),
            )
            self.calculation_process = Process(target=_run_opt, args=opt_params, name='opt_final')
            self.calculation_process.start()
        else:
            sp_params = (
                atoms,
                self.final_traj_file(),
                self.final_xyz_file(),
            )
            self.calculation_process = Process(target=_run_single_point, args=sp_params, name='sp_final')
            self.calculation_process.start()
        return True

    def run_g16_init_guess(self, prev_number: Optional[int] = None) -> bool:
        if self.calculation_process is not None:
            return False

        # Settings to run g16
        G16Calculator.set_environment(g16_root=config.G16_ROOT)
        if config.G16_SCRATCH_DIR is None:
            scratch_dir = self.work_dir / 'g16data'
        else:
            scratch_dir = Path(config.G16_SCRATCH_DIR).absolute()

        if prev_number is None:
            prev_number = self.current_final_neb_number()
        if prev_number == 0:
            prev_file = self.initial_path_traj_file()
        else:
            prev_file = self.neb_path_traj_file(prev_number)
        nodes = ase.io.read(str(prev_file) + '@-{:}:'.format(self.num_images+2))

        # prepare all gjf files for run
        prev_dir = Path(os.getcwd())
        try:
            g16_dir = self.work_dir / 'g16data'
            if not g16_dir.exists():
                g16_dir.mkdir(parents=True)
            os.chdir(g16_dir)
            gjf_list = []
            for i, atoms in enumerate(nodes):
                job_name = 'node{:0>3d}'.format(i)

                # remove all previous files
                suffix_list = [
                    '.gjf',
                    '_init_guess.gjf'
                    '.log'
                    '_init_guess.log',
                    '.out',
                    '_init_guess.out',
                ]
                for suffix in suffix_list:
                    remove(Path(job_name + suffix))

                # prepare gjf file for init guess
                with open(job_name + '_init_guess.gjf', mode='w', encoding='utf-8', newline='\n') as f:
                    f.writelines(self._prepare_gjf_data_for_guess(atoms, job_name))
                gjf_list.append(job_name + '_init_guess.gjf')
        finally:
            os.chdir(prev_dir)

        # file lists to be removed after g16 run
        remove_files = []
        for i in range(len(nodes)):
            job_name = 'node{:0>3d}'.format(i)
            for suffix in ['_init_guess.rwf', '_init_guess.int', '_init_guess.d2e']:
                remove_files.append(Path(scratch_dir / (job_name + suffix)))

        # Run g16 in a separated process
        self.calculation_process = Process(target=_run_g16_init_guess,
                                           args=(self.neb_parallel, gjf_list, g16_dir, remove_files),
                                           name='g16_init_guess')
        self.calculation_process.start()

    def check(self) -> int:
        """
        Return status code:
        Calculation is not running: -1
        Calculation is finished: 0
        Calculation is running: 1
        """
        if self.calculation_process is None:
            return -1
        elif self.calculation_process.is_alive():
            return 1
        else:
            self.calculation_process.join()
            self.calculation_process.close()
            self.calculation_process = None
            return 0

    def terminate(self) -> None:
        self.calculation_process.terminate()
        self.calculation_process.kill()
        self.calculation_process.join()
        self.calculation_process.close()
        self.calculation_process = None

    def current_calculation_log_file(self) -> Optional[Path]:
        if self.calculation_process is None:
            return None
        elif self.calculation_process.name.startswith('neb'):
            number = int(self.calculation_process.name[3:])
            return self.neb_path_log_file(number)
        elif self.calculation_process.name == 'opt_init':
            return self.init_log_file()
        elif self.calculation_process.name == 'opt_final':
            return self.final_log_file()
        else:
            return None

    def current_calculation_job_name(self) -> Optional[str]:
        """
        None
        opt_init
        sp_init
        opt_final
        sp_final
        g16_init_guess
        neb@ (@ = 1,2,3...)
        """
        if self.calculation_process is None:
            return None
        else:
            return self.calculation_process.name


def _run_neb(neb: NEB,
             optimizer: str,
             trajectory_file: Path,
             log_file: Path,
             xyz_file: Path,
             optimized_xyz_file: Path,
             fmax: float,
             steps: int,
             num_nodes: int) -> None:
    # optimizer settings and run as a subprocess
    if optimizer.upper() == 'FIRE':
        # noinspection PyTypeChecker
        optimizer = FIRE(neb, trajectory=str(trajectory_file), logfile=str(log_file))
        optimizer.run(fmax=fmax, steps=steps)

    elif optimizer.upper() == 'LBFGS':
        optimizer = LBFGS(neb, trajectory=str(trajectory_file), logfile=str(log_file))
        optimizer.run(fmax=fmax, steps=steps)

    elif optimizer.upper() == 'LBFGSLINESEARCH':
        optimizer = LBFGSLineSearch(neb, trajectory=trajectory_file, logfile=log_file)
        optimizer.run(fmax=fmax, steps=steps)

    elif optimizer.upper() == 'COMPOSITE':
        optimizer = FIRE(neb, trajectory=str(trajectory_file), logfile=str(log_file))
        optimizer.run(fmax=max(0.1, fmax), steps=steps)
        optimizer = LBFGS(neb, trajectory=str(trajectory_file), logfile=str(log_file))
        optimizer.run(fmax=fmax, steps=steps)

    else:
        raise RuntimeError('Optimizer is invalid [FIRE, LBFGS, LBFGSLineSearch, Composite]')

    # Save xyz
    traj = NEBResult(trajectory_file, num_nodes=num_nodes)
    traj.save_xyz(xyz_file, energy_unit='eV')
    traj.save_xyz(optimized_xyz_file, iteration=-1, energy_unit='eV')


def _run_opt(atoms: Atoms,
             trajectory_file: Path,
             log_file: Path,
             xyz_file: Path) -> None:
    optimizer = BFGSLineSearch(atoms,
                               logfile=str(log_file),
                               trajectory=str(trajectory_file))
    optimizer.run(fmax=config.PREOPT_FMAX, steps=config.PREOPT_STEPS)
    traj = SingleTrajectory(trajectory_file)
    traj.save_xyz(xyz_file, energy_unit='eV')


def _run_single_point(atoms: Atoms,
                      trajectory_file: Path,
                      xyz_file: Path) -> None:
    atoms.get_potential_energy()
    ase.io.write(trajectory_file, atoms)
    traj = SingleTrajectory(trajectory_file)
    traj.save_xyz(xyz_file, energy_unit='eV')


def _run_g16_init_guess(num_process: int, gjf_list: List[str], g16_dir: Path, remove_files: List[Path]) -> None:
    prev_dir = Path(os.getcwd())
    try:
        os.chdir(g16_dir)
        if num_process > 1:
            with Pool(num_process) as pool:
                pool.map(run_g16, gjf_list)
        else:
            for gjf in gjf_list:
                run_g16(gjf)
    except:
        raise
    finally:
        os.chdir(prev_dir)
        for file in remove_files:
            remove(file, ignore_errors=True)
