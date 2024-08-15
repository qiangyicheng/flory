import numpy as np


class FreeEnergyBase:
    def __init__(self, num_comp: int, size: np.ndarray | None = None):
        self.num_comp = num_comp
        if size is None:
            self.size = np.ones(num_comp)
        else:
            self.size = np.array(size)
            if self.size.shape != (self.num_comp,):
                raise ValueError("The sizes vector must be have the length as chis")
            if np.any(self.size <= 0):
                raise ValueError("The sizes vector must be all positive.")

    def make_interaction(self):
        raise NotImplementedError

    def free_energy_density(self, phis: np.ndarray) -> np.ndarray:
        """returns free energy for a given composition

        Args:
            phis (:class:`numpy.ndarray`): The composition of the phase(s)
        """
        raise NotImplementedError

    def chemical_potentials(self, phis: np.ndarray) -> np.ndarray:
        """returns chemical potentials for a given composition"""
        raise NotImplementedError

    def hessian(self, phis: np.ndarray) -> np.ndarray:
        """returns Hessian for the given composition"""
        raise NotImplementedError

    def pressure(self, phis: np.ndarray) -> np.ndarray:
        """returns pressure for a given composition"""
        f = self.free_energy_density(phis)
        mus = self.chemical_potentials(phis)
        return np.einsum("...i,...i->...", phis, mus) - f  # type: ignore

    def unstable_modes(self, phis: np.ndarray) -> int:
        """returns the number of unstable modes"""
        eigenvalues = np.linalg.eigvalsh(self.hessian(phis))
        return int(np.sum(eigenvalues < 0))

    def is_stable(self, phis: np.ndarray) -> bool:
        """checks whether a given composition is (linearly) stable"""
        return self.unstable_modes(phis) == 0
