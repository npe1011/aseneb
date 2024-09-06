from multiprocessing import Pool

import numpy as np
from ase.atoms import Atoms
from ase.mep import NEB
from ase.mep.neb import NEBState
from ase.build import minimize_rotation_and_translation
from ase.optimize.precon import Precon, PreconImages


class PalNEB(NEB):
    """
    Multiprocessing NEB by pool executor for force calculation
    Partially copied and modified from: https://gitlab.com/ase/ase/-/blob/master/ase/mep/neb.py
    """
    def __init__(self, images, k=0.1, climb=False, num_processes=1,
                 remove_rotation_and_translation=False, method='aseneb', precon=None, **kwargs):

        self.num_processes = num_processes
        super().__init__(
            images, k=k, climb=climb, parallel=False,
            remove_rotation_and_translation=remove_rotation_and_translation, world=None,
            method=method, allow_shared_calculator=False,
            precon=precon, **kwargs
        )

    def get_forces(self):
        """
        Evaluate and return the forces with multiprocessing.
        This method is the modified version of the original ASE code
        https://gitlab.com/ase/ase/-/blob/master/ase/mep/neb.py
        """
        images = self.images

        if not self.allow_shared_calculator:
            calculators = [image.calc for image in images
                           if image.calc is not None]
            if len(set(calculators)) != len(calculators):
                msg = ('One or more NEB images share the same calculator.  '
                       'Each image must have its own calculator.  '
                       'You may wish to use the ase.mep.SingleCalculatorNEB '
                       'class instead, although using separate calculators '
                       'is recommended.')
                raise ValueError(msg)

        forces = np.empty(((self.nimages - 2), self.natoms, 3))
        energies = np.empty(self.nimages)

        if self.remove_rotation_and_translation:
            for i in range(1, self.nimages):
                minimize_rotation_and_translation(images[i - 1], images[i])

        # Modification for force calculations
        if self.method != 'aseneb':
            with Pool(processes=self.num_processes) as pool:
                energy_forces = pool.map(_calc_energy_and_forces, images)
            energies[0] = energy_forces[0][0]
            energies[-1] = energy_forces[-1][0]
            for i in range(1, self.nimages - 1):
                forces[i-1] = energy_forces[i][1]
                energies[i] = energy_forces[i][0]

        else:
            with Pool(processes=self.num_processes) as pool:
                energy_forces = pool.map(_calc_energy_and_forces, images[1:-1])
            for i in range(1, self.nimages - 1):
                forces[i-1] = energy_forces[i-1][1]
                energies[i] = energy_forces[i-1][0]

        # if this is the first force call, we need to build the preconditioners
        if self.precon is None or isinstance(self.precon, (str, Precon, list)):
            self.precon = PreconImages(self.precon, images)

        # apply preconditioners to transform forces
        # for the default IdentityPrecon this does not change their values
        precon_forces = self.precon.apply(forces, index=slice(1, -1))

        # Save for later use in iterimages:
        self.energies = energies
        self.real_forces = np.zeros((self.nimages, self.natoms, 3))
        self.real_forces[1:-1] = forces

        state = NEBState(self, images, energies)

        # Can we get rid of self.energies, self.imax, self.emax etc.?
        self.imax = state.imax
        self.emax = state.emax

        spring1 = state.spring(0)

        self.residuals = []
        for i in range(1, self.nimages - 1):
            spring2 = state.spring(i)
            tangent = self.neb_method.get_tangent(state, spring1, spring2, i)

            # Get overlap between full PES-derived force and tangent
            tangential_force = np.vdot(forces[i - 1], tangent)

            # from now on we use the preconditioned forces (equal for precon=ID)
            imgforce = precon_forces[i - 1]

            if i == self.imax and self.climb:
                """The climbing image, imax, is not affected by the spring
                   forces. This image feels the full PES-derived force,
                   but the tangential component is inverted:
                   see Eq. 5 in paper II."""
                if self.method == 'aseneb':
                    tangent_mag = np.vdot(tangent, tangent)  # For normalizing
                    imgforce -= 2 * tangential_force / tangent_mag * tangent
                else:
                    imgforce -= 2 * tangential_force * tangent
            else:
                self.neb_method.add_image_force(state, tangential_force,
                                                tangent, imgforce, spring1,
                                                spring2, i)
                # compute the residual - with ID precon, this is just max force
                residual = self.precon.get_residual(i, imgforce)
                self.residuals.append(residual)

            spring1 = spring2

        return precon_forces.reshape((-1, 3))


def _calc_energy_and_forces(atoms: Atoms) -> tuple:
    forces = atoms.get_forces()
    energy = atoms.get_potential_energy()
    return energy, forces
