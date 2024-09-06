import os
from typing import List, Union, Optional
import json
import shutil
from pathlib import Path
from collections import OrderedDict
from multiprocessing import Pool

import numpy as np
import ase.io
from ase.atoms import Atoms
from ase.mep import NEB
from ase.optimize import LBFGS, LBFGSLineSearch, BFGSLineSearch, FIRE
from ase.calculators.calculator import Calculator

from aseneb.palneb import PalNEB
from aseneb.xtbcalc import XTBParams, XTBCalculator
from aseneb.g16calc import G16Calculator, run_g16
from aseneb.ase_result import NEBResult, SingleTrajectory
from aseneb.utils import atoms_to_text, remove
import config


class NEBProject:

    def __init__(self, json_file: Union[str, Path, None] = None):

        self.work_dir: Path = Path.cwd()
        self.project_name: str = config.DEFAULT_PROJECT_NAME

        self.opt_init: bool = config.DEFAULT_OPT_INIT
        self.opt_final: bool = config.DEFAULT_OPT_FINAL

        self.num_images: int = config.DEFAULT_NUM_IMAGES
        self.interpolation_method: str = config.DEFAULT_INTERPOLATION_METHOD

        self.neb_k: float = config.DEFAULT_NEB_K
        self.neb_climb: bool = config.DEFAULT_NEB_CLIMB
        self.neb_method: str = config.DEFAULT_NEB_METHOD
        self.neb_optimizer: str = config.DEFAULT_NEB_OPTIMIZER
        self.neb_fmax: float = config.DEFAULT_NEB_FMAX
        self.neb_steps:int = config.DEFAULT_NEB_STEPS
        self.neb_parallel: int = config.DEFAULT_NEB_PARALLEL

        self.calculator_type = config.DEFAULT_CALCULATOR_TYPE

        self.xtb_gfn: str = config.DEFAULT_XTB_GFN
        self.xtb_solvation: Optional[str] = config.DEFAULT_XTB_SOLVATION
        self.xtb_solvent: Optional[str] = config.DEFAULT_XTB_SOLVENT
        self.xtb_uhf: int = config.DEFAULT_XTB_UHF
        self.xtb_charge: int = config.DEFAULT_XTB_CHARGE
        self.xtb_cpu: int = config.DEFAULT_XTB_CPU
        self.xtb_memory_per_cpu: str = config.DEFAULT_XTB_MEMORY

        self.g16_cpu: int = config.DEFAULT_G16_CPU
        self.g16_memory: str = config.DEFAULT_G16_MEMORY
        self.g16_guess_additional_keywords: str = config.DEFAULT_G16_GUESS_ADDITIONAL_KEYWORDS

        if json_file is not None:
            self.read_json(json_file)

    def read_json(self, json_file: Union[str, Path]) -> None:
        self.work_dir = Path(json_file).absolute().parent
        self.project_name = Path(json_file).absolute().stem
        with open(json_file, mode='r') as f:
            data = json.load(f)
            self.num_images = data['num_images']
            self.opt_init = data['opt_init']
            self.opt_final = data['opt_final']
            self.interpolation_method = data['interpolation_method']
            self.neb_k = data['neb_k']
            self.neb_climb = data['neb_climb']
            self.neb_method = data['neb_method']
            self.neb_optimizer = data['neb_optimizer']
            self.neb_fmax = data['neb_fmax']
            self.neb_steps = data['neb_steps']
            self.neb_parallel = data['neb_parallel']
            self.calculator_type = data['calculator_type']
            self.xtb_gfn = data['xtb_gfn']
            self.xtb_solvation = data['xtb_solvation']
            self.xtb_solvent = data['xtb_solvent']
            self.xtb_uhf = data['xtb_uhf']
            self.xtb_charge = data['xtb_charge']
            self.xtb_cpu = data['xtb_cpu']
            self.xtb_memory_per_cpu = data['xtb_memory_per_cpu']
            self.g16_cpu = data['g16_cpu']
            self.g16_memory = data['g16_memory']
            self.g16_guess_additional_keywords = data['g16_guess_additional_keywords']

    def save_json(self) -> None:
        json_file = self.work_dir / (self.project_name + '.json')
        data = OrderedDict()
        data['num_images'] = self.num_images
        data['opt_init'] = self.opt_init
        data['opt_final'] = self.opt_final
        data['interpolation_method'] = self.interpolation_method
        data['neb_k'] = self.neb_k
        data['neb_climb'] = self.neb_climb
        data['neb_method'] = self.neb_method
        data['neb_optimizer'] = self.neb_optimizer
        data['neb_fmax'] = self.neb_fmax
        data['neb_steps'] = self.neb_steps
        data['neb_parallel'] = self.neb_parallel
        data['calculator_type'] = self.calculator_type
        data['xtb_gfn'] = self.xtb_gfn
        data['xtb_solvation'] = self.xtb_solvation
        data['xtb_solvent'] = self.xtb_solvent
        data['xtb_uhf'] = self.xtb_uhf
        data['xtb_charge'] = self.xtb_charge
        data['xtb_cpu'] = self.xtb_cpu
        data['xtb_memory_per_cpu'] = self.xtb_memory_per_cpu
        data['g16_cpu'] = self.g16_cpu
        data['g16_memory'] = self.g16_memory
        data['g16_guess_additional_keywords'] = self.g16_guess_additional_keywords

        with open(json_file, mode='w') as f:
            json.dump(data, f, indent=2)

    def load_init_structure(self, file: Union[str, Path]) -> None:
        atoms = ase.io.read(file, index=-1)
        calc = self.get_calculator()
        if self.calculator_type == 'g16':
            calc.job_name = 'init'
        atoms.set_calculator(calc)
        remove(self.init_traj_file())
        remove(self.init_log_file())
        remove(self.init_xyz_file())
        if self.opt_init:
            optimizer = BFGSLineSearch(atoms,
                                       logfile=str(self.init_log_file()),
                                       trajectory=str(self.init_traj_file()))
            optimizer.run(fmax=config.PREOPT_FMAX, steps=config.PREOPT_STEPS)

        else:
            atoms.get_potential_energy()
            ase.io.write(self.init_traj_file(), atoms)
        traj = SingleTrajectory(self.init_traj_file())
        traj.save_xyz(self.init_xyz_file(), energy_unit='eV')

    def load_final_structure(self, file: Union[str, Path]) -> None:
        atoms = ase.io.read(file, index=-1)
        calc = self.get_calculator()
        if self.calculator_type == 'g16':
            calc.job_name = 'final'
        atoms.set_calculator(calc)
        remove(self.final_traj_file())
        remove(self.final_log_file())
        remove(self.final_xyz_file())
        if self.opt_final:
            optimizer = BFGSLineSearch(atoms,
                                       logfile=str(self.final_log_file()),
                                       trajectory=str(self.final_traj_file()))
            optimizer.run(fmax=config.PREOPT_FMAX, steps=config.PREOPT_STEPS)
        else:
            atoms.get_potential_energy()
            ase.io.write(self.final_traj_file(), atoms)
        traj = SingleTrajectory(self.final_traj_file())
        traj.save_xyz(self.final_xyz_file(), energy_unit='eV')

    def interpolate(self) -> None:
        remove(self.initial_path_traj_file())
        remove(self.initial_path_xyz_file())

        # Check and read init and final trajectory files.
        if not self.init_traj_file().exists():
            raise RuntimeError('Initial structure is not loaded.')
        if not self.final_traj_file().exists():
            raise RuntimeError('Final structure is not loaded.')
        init_atoms = ase.io.read(self.init_traj_file(), index=-1)
        final_atoms = ase.io.read(self.final_traj_file(), index=-1)

        # Check init and final structures have same atoms
        try:
            assert np.all(init_atoms.get_atomic_numbers() == final_atoms.get_atomic_numbers())
        except:
            raise RuntimeError('Initial and Final structures must have the same atoms in the same order.')

        # prepare nodes and interpolation
        nodes = [init_atoms]
        for i in range(self.num_images):
            nodes.append(init_atoms.copy())
        nodes += [final_atoms]
        neb = NEB(images=nodes, k=self.neb_k, climb=self.neb_climb, remove_rotation_and_translation=True)
        if self.interpolation_method.lower() == 'idpp':
            neb.interpolate(method='idpp')
        elif self.interpolation_method.lower() == 'linear':
            neb.interpolate()
        else:
            raise RuntimeError('Interpolation method is invalid. [linear or idpp]')

        ase.io.write(self.initial_path_traj_file(), nodes)
        traj = SingleTrajectory(self.initial_path_traj_file())
        traj.save_xyz(self.initial_path_xyz_file(), energy_unit='eV')

    def run_neb(self, prev_number: Optional[int] = None) -> bool:
        """
        Run NEB calculation from the prev_number path. If None, the latest one is read and rerun NEB optimization.
        """

        if prev_number is None:
            prev_number = self.current_final_neb_number()
        if prev_number == 0:
            prev_file = self.initial_path_traj_file()
        else:
            prev_file = self.neb_path_traj_file(prev_number)

        if not prev_file.exists():
            raise RuntimeError(str(prev_file) + ' is not found.')

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

        # optimizer settings and run
        if self.neb_optimizer.upper() == 'FIRE':
            optimizer = FIRE(neb, trajectory=str(trajectory_file), logfile=str(log_file))
            optimizer.run(fmax=self.neb_fmax, steps=self.neb_steps)
        elif self.neb_optimizer.upper() == 'LBFGS':
            optimizer = LBFGS(neb, trajectory=str(trajectory_file), logfile=str(log_file))
            optimizer.run(fmax=self.neb_fmax, steps=self.neb_steps)
        elif self.neb_optimizer.upper() == 'LBFGSLINESEARCH':
            optimizer = LBFGSLineSearch(neb, trajectory=str(trajectory_file), logfile=str(log_file))
            optimizer.run(fmax=self.neb_fmax, steps=self.neb_steps)
        elif self.neb_optimizer.upper() == 'COMPOSITE':
            optimizer = FIRE(neb, trajectory=str(trajectory_file), logfile=str(log_file))
            optimizer.run(fmax=max(0.1, self.neb_fmax), steps=self.neb_steps)
            optimizer = LBFGS(neb, trajectory=str(trajectory_file), logfile=str(log_file))
            optimizer.run(fmax=self.neb_fmax, steps=self.neb_steps)
        else:
            raise RuntimeError('Optimizer is invalid [FIRE, LBFGS, LBFGSLineSearch, Composite]')

        # Save xyz
        traj = NEBResult(trajectory_file, num_nodes=self.num_images+2)
        traj.save_xyz(xyz_file, energy_unit='eV')
        traj.save_xyz(optimized_xyz_file, iteration=-1, energy_unit='eV')

        return True

    def run_g16_init_guess(self, prev_number: Optional[int] = None) -> bool:
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

        prev_dir = Path(os.getcwd())
        try:
            os.chdir(self.work_dir / 'g16data')
            # prepare all gjf files for run
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

            # Run g16
            if self.neb_parallel > 1:
                with Pool(self.neb_parallel) as pool:
                    pool.map(run_g16, gjf_list)
            else:
                for gjf in gjf_list:
                    run_g16(gjf)

            return True

        except:
            raise

        finally:
            os.chdir(prev_dir)
            for i in range(len(nodes)):
                job_name = 'node{:0>3d}'.format(i)
                for suffix in ['_init_guess.rwf', '_init_guess.int', '_init_guess.d2e']:
                    file = Path(scratch_dir / (job_name + suffix))
                    remove(file, ignore_errors=True)

    def clear_all_results(self):
        # init
        remove(self.init_traj_file())
        remove(self.init_xyz_file())
        remove(self.init_log_file())
        # final
        remove(self.final_traj_file())
        remove(self.final_xyz_file())
        remove(self.final_log_file())
        # interpolate
        remove(self.initial_path_traj_file())
        remove(self.initial_path_xyz_file())
        # neb
        for i in range(1, self.current_final_neb_number() + 1):
            remove(self.neb_path_traj_file(i))
            remove(self.neb_path_xyz_file(i))
            remove(self.neb_path_log_file(i))
            remove(self.neb_path_optimized_xyz_file(i))
        # g16 chk files
        g16_dir = self.work_dir / 'g16data'
        if g16_dir.exists():
            shutil.rmtree(g16_dir)

    def get_calculator(self) -> Calculator:

        if self.calculator_type.lower() == 'xtb':
            xtb_params = XTBParams(method=self.xtb_gfn,
                                    charge=self.xtb_charge,
                                    uhf=self.xtb_uhf,
                                    solvation=self.xtb_solvation,
                                    solvent=self.xtb_solvent)
            calculator = XTBCalculator(xtb_params=xtb_params,
                                       workdir=self.work_dir,
                                       xtb_bin=config.XTB_BIN,
                                       xtb_param_dir=config.XTB_PARAM_DIR,
                                       omp_num_threads=self.xtb_cpu,
                                       omp_stacksize=self.xtb_memory_per_cpu)

        elif self.calculator_type.lower() == 'g16':
            G16Calculator.set_environment(g16_root=config.G16_ROOT)
            calculator = G16Calculator(template_file=self.work_dir / config.G16_TEMPLATE_FILE_NAME,
                                       job_name='aserun.gjf',
                                       num_procs=self.g16_cpu,
                                       memory=self.g16_memory,
                                       workdir=self.work_dir / 'g16data')
            (self.work_dir / 'g16data').mkdir(exist_ok=True)
            return calculator

        else:
            raise RuntimeError('Given calculator type is invalid.')

        return calculator

    def get_neb_result(self, number: Optional[int] = None) -> NEBResult:
        target_file = self.neb_path_traj_file(number)
        if not target_file.exists():
            raise FileNotFoundError('The NEB result not found:', str(target_file))
        res = NEBResult(target_file, num_nodes=self.num_images+2)
        init = SingleTrajectory(self.init_traj_file())
        final = SingleTrajectory(self.final_traj_file())
        res.complete_energy(0, init.energies[-1])
        res.complete_energy(-1, final.energies[-1])
        return res

    def init_traj_file(self) -> Path:
        return self.work_dir / '{}_init.traj'.format(self.project_name)

    def init_xyz_file(self) -> Path:
        return self.work_dir / '{}_init.xyz'.format(self.project_name)

    def init_log_file(self) -> Path:
        return self.work_dir / '{}_init.log'.format(self.project_name)

    def final_traj_file(self) -> Path:
        return self.work_dir / '{}_final.traj'.format(self.project_name)

    def final_xyz_file(self) -> Path:
        return self.work_dir / '{}_final.xyz'.format(self.project_name)

    def final_log_file(self) -> Path:
        return self.work_dir / '{}_final.log'.format(self.project_name)

    def initial_path_traj_file(self) -> Path:
        return self.work_dir / '{}_initial_path.traj'.format(self.project_name)

    def initial_path_xyz_file(self) -> Path:
        return self.work_dir / '{}_initial_path.xyz'.format(self.project_name)

    def neb_path_traj_file(self, number: Optional[int] = None) -> Path:
        if number is None:
            number = self.current_final_neb_number()
        return self.work_dir / '{}_neb_path_{}.traj'.format(self.project_name, number)

    def neb_path_xyz_file(self, number: Optional[int] = None) -> Path:
        if number is None:
            number = self.current_final_neb_number()
        return self.work_dir / '{}_neb_path_{}.xyz'.format(self.project_name, number)

    def neb_path_optimized_xyz_file(self, number: Optional[int] = None) -> Path:
        if number is None:
            number = self.current_final_neb_number()
        return self.work_dir / '{}_neb_path_optimized_{}.xyz'.format(self.project_name, number)

    def neb_path_log_file(self, number: int) -> Path:
        return self.work_dir / '{}_neb_path_{}.log'.format(self.project_name, number)

    def get_all_neb_traj_files(self) -> List[Path]:
        neb_traj_list = list(self.work_dir.glob('{}_neb_path_*.traj'.format(self.project_name)))
        neb_traj_list.sort(key=lambda x: int(str(x)[:-5].split('_')[-1]))
        return neb_traj_list

    def current_final_neb_number(self) -> int:
        neb_traj_list = self.work_dir.glob('{}_neb_path_*.traj'.format(self.project_name))
        max_number = 0
        for file in neb_traj_list:
            number = int(str(file)[:-5].split('_')[-1])
            max_number = max(max_number, number)
        return max_number

    def _prepare_gjf_data_for_guess(self, atoms: Atoms, job_name) -> List[str]:

        if config.G16_SCRATCH_DIR is None:
            scratch_dir = self.work_dir / 'g16data'
        else:
            scratch_dir = Path(config.G16_SCRATCH_DIR).absolute()

        template_file = self.work_dir / config.G16_TEMPLATE_FILE_NAME

        data = [
            '%nprocshared={:}\n'.format(self.g16_cpu),
            '%mem={:}\n'.format(self.g16_memory),
            '%rwf={:}_init_guess.rwf\n'.format(scratch_dir / job_name),
            '%int={:}_init_guess.int\n'.format(scratch_dir / job_name),
            '%d2e={:}_init_guess.d2e\n'.format(scratch_dir / job_name),
            '%nosave\n',
            '%chk={:}.chk\n'.format(job_name)
        ]

        with template_file.open() as f:
            for line in f.readlines():
                if line.strip().startswith('%'):
                    continue
                elif line.strip().startswith('#'):
                    data.append(line.rstrip() + ' nosymm ' + self.g16_guess_additional_keywords.strip() + '\n')
                elif line.strip().lower().startswith('@'):
                    data.extend(atoms_to_text(atoms))
                    continue
                else:
                    data.append(line)

        return data
