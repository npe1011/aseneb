from typing import Union, Tuple, List
from pathlib import Path

import ase.io
import numpy as np
from matplotlib import pyplot as plt

from aseneb.utils import atoms_to_text, energy_conversion


class NEBResult:
    def __init__(self, trajectory_file: Union[str, Path], num_nodes: int):
        self.num_iteration: int = 0
        self.num_nodes: int = 0
        self.atoms_list_list: List[List[ase.atoms.Atoms]] = []
        self.energies_list: List[np.ndarray] = []  # in eV (ase default)

        self.num_nodes = num_nodes

        # read full trajectory file and split into List[Atoms]
        atoms_list = ase.io.read(str(trajectory_file) + '@:')
        self.num_iteration = len(atoms_list) // self.num_nodes
        for i in range(self.num_iteration):
            self.atoms_list_list.append(atoms_list[i*num_nodes:(i+1)*num_nodes])

        # prepare energies list (nan for atoms with no energy)
        for i in range(self.num_iteration):
            energies = np.zeros(self.num_nodes, dtype=float)
            for n, atoms in enumerate(self.atoms_list_list[i]):
                try:
                    e = atoms.get_potential_energy()
                except:
                    e = np.nan
                energies[n] = e
            self.energies_list.append(energies)

    def is_energy_completed(self) -> bool:
        """
        Check if there are in no np.nan for all nodes in all iteration.
        """
        for energies in self.energies_list:
            if np.any(np.isnan(energies)):
                return False
        return True

    def complete_energy(self, index: int, energy: float, overwrite=False) -> None:
        """
        Fulfill energy of node of index for all trajectory, if np.nan.
        Overwrite=true forces overwrite energy even if not np.nan
        """
        for energies in self.energies_list:
            if np.isnan(energies[index]) or overwrite:
                energies[index] = energy

    def plot_mep_all(self, energy_unit:str = 'kcal/mol') -> None:
        """
        plot energy profile for all iteration with color map
        """
        if not self.is_energy_completed():
            raise RuntimeError('Some energy information is missing. Check the initial and final Atoms.')

        cmap = plt.get_cmap('Blues')
        xs = np.array(range(self.num_nodes), dtype=int)
        for i, energies in enumerate(self.energies_list):
            plt.plot(xs,
                     energy_conversion(energies - energies[0], energy_unit),
                     color=cmap(i + 1 / self.num_iteration))

        plt.xlabel('# Node')
        plt.ylabel('E ({})'.format(energy_unit))
        plt.title('NEB Energy Profile (All Iteration)')
        plt.tight_layout()
        plt.show()

    def plot_mep(self, iteration: int = -1, energy_unit: str = 'kcal/mol') -> None:
        """
        plot energy profile of the given iteration
        """
        if not self.is_energy_completed():
            raise RuntimeError('Some energy information is missing. Check the initial and final Atoms.')

        xs = np.array(range(self.num_nodes), dtype=int)
        energies = energy_conversion(self.energies_list[iteration] - self.energies_list[iteration][0], energy_unit)
        plt.plot(xs, energies, marker='.')
        plt.xlabel('# Node')
        plt.ylabel('E ({})'.format(energy_unit))
        plt.title('NEB Energy Profile (Last Iteration)')
        plt.tight_layout()
        plt.show()

    def get_barrier(self, iteration: int = -1, energy_unit:str = 'kcal/mol') -> float:
        if not self.is_energy_completed():
            raise RuntimeError('Some energy information is missing. Check the initial and final Atoms.')
        energies = self.energies_list[iteration]
        return float(energy_conversion(np.max(energies) - energies[0], energy_unit))

    def get_highest_node_index(self, iteration: int = -1) -> int:
        return int(np.argmax(self.energies_list[iteration]))

    def get_reaction_energy_change(self, iteration: int = -1, energy_unit:str = 'kcal/mol') -> float:
        if not self.is_energy_completed():
            raise RuntimeError('Some energy information is missing. Check the initial and final Atoms.')
        energies = self.energies_list[iteration]
        return float(energy_conversion(energies[-1] - energies[0], energy_unit))

    def save_xyz(self,
                 xyz_file: Union[str, Path],
                 iteration: Union[int, Tuple[int, int], None] = None,
                 node: Union[int, Tuple[int, int], None] = None,
                 energy_unit:str = 'eV') -> None:

        if type(iteration) is int:
            iteration_range = range(iteration, iteration + 1)
        elif type(iteration) is tuple or type(iteration) is list:
            iteration_range = range(iteration[0], iteration[1])
        else:
            iteration_range = range(0, self.num_iteration)

        if type(node) is int:
            node_range = range(node, node + 1)
        elif type(node) is tuple or type(node) is list:
            node_range = range(node[0], node[1])
        else:
            node_range = range(0, self.num_nodes)

        xyz_data = []
        for i in iteration_range:
            for n in node_range:
                atoms = self.atoms_list_list[i][n]
                xyz_data.append('{}\n'.format(len(atoms)))
                if self.energies_list[i][n] is None:
                    energy_string = 'nan'
                else:
                    energy_string = '{:>12.8f} ({:})'.format(self.energies_list[i][n], energy_unit)

                i_ = i
                n_ = n
                if i < 0:
                    i_ = i + self.num_iteration
                if n < 0:
                    n_ = n + self.num_nodes

                title = '#ITR {:}; #NODE {:}; E = {:}\n'.format(i_, n_, energy_string)
                xyz_data.append(title)
                xyz_data.extend(atoms_to_text(atoms))

        with Path(xyz_file).open(mode='w', encoding='utf-8', newline='\n') as f:
            f.writelines(xyz_data)


class SingleTrajectory:
    def __init__(self, trajectory_file: Union[str, Path]):
        self.num_nodes: int = 0
        self.atoms_list: List[List[ase.atoms.Atoms]] = []
        self.energies: np.ndarray = np.zeros(1)  # in eV (ase default)

        # read trajectory file
        atoms_list = ase.io.read(str(trajectory_file) + '@:')
        if type(atoms_list) is list:
            self.atoms_list = atoms_list
            self.num_nodes = len(atoms_list)
        else:
            self.atoms_list = [atoms_list]
            self.num_nodes = 1

        # prepare energies (nan for atoms with no energy)
        energies = np.zeros(self.num_nodes, dtype=float)
        for n, atoms in enumerate(self.atoms_list):
            try:
                e = atoms.get_potential_energy()
            except:
                e = np.nan
            energies[n] = e
        self.energies = energies

    def plot_energies(self, energy_unit: str = 'kcal/mol') -> None:
        """
        plot energy profile
        """
        if self.num_nodes <= 1:
            raise RuntimeError('Only 1 structure exists.')

        if not self.is_energy_completed():
            raise RuntimeError('Some energy information is missing. Check the initial and final Atoms.')

        xs = np.array(range(self.num_nodes), dtype=int)
        energies = energy_conversion(self.energies - self.energies[0], energy_unit)
        plt.plot(xs, energies, marker='.')
        plt.xlabel('# Node')
        plt.ylabel('E ({})'.format(energy_unit))
        plt.tight_layout()
        plt.show()

    def is_energy_completed(self) -> bool:
        """
        Check if there are in no np.nan for all nodes.
        """
        return not np.any(np.isnan(self.energies))

    def save_xyz(self, xyz_file: Union[str, Path], energy_unit: str = 'eV') -> None:
        xyz_data = []
        for n in range(self.num_nodes):
            atoms = self.atoms_list[n]
            xyz_data.append('{}\n'.format(len(atoms)))
            if self.energies[n] is None:
                energy_string = 'nan'
            else:
                energy_string = '{:>12.8f} ({:})'.format(self.energies[n], energy_unit)
            title = '#NODE {:}; E = {:}\n'.format(n, energy_string)
            xyz_data.append(title)
            xyz_data.extend(atoms_to_text(atoms))

        with Path(xyz_file).open(mode='w', encoding='utf-8', newline='\n') as f:
            f.writelines(xyz_data)





