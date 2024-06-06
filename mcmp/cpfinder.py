"""

.. autosummary::
   :nosignatures:

.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import logging
from tqdm.auto import tqdm
from datetime import datetime

import numba as nb
import numpy as np


@nb.njit
def find_coexisting_phases_impl(
    phi_means: np.ndarray,
    chis: np.ndarray,
    sizes: np.ndarray,
    *,
    omegas: np.ndarray,
    Js: np.ndarray,
    phis: np.ndarray,
    steps_inner: int,
    acceptance_Js: float,
    Js_step_upperbound: float,
    acceptance_omega: float,
    kill_threshold: float,
    revive_tries: int,
    revive_scaler: float,
    rng: np.random.Generator,
) -> tuple[float, float, float, int, bool]:
    """
    The implementation of the core algorithm for finding coexisting states of
    Flory-Huggins system.

    Args:
        phi_means (np.ndarray):
            input, the average volume fraction of all the components of the system. 1D
            array with size of num_comps. Note that the volume fraction of the solvent is
            included as well, therefore the sum of this array must be unity, which is not
            checked by this function and should be guaranteed externally.
        chis (np.ndarray):
            input, the interaction matrix. 2D array with size of num_comps-by-num_comps.
            This chi matrix should be the full chi matrix of the system, including the
            solvent component. Note that the symmetry is not checked, which should be
            guaranteed externally.
        sizes (np.ndarray):
            input, the relative molecule volumes of the components. 1D array with size of
            num_comps. This sizes vector should be the full sizes vector of the system,
            including the solvent component.
        omegas (np.ndarray):
            initialization and output, the conjugate fields of the volume fractions. 2D
            array with size of num_comps-by-num_phases. Note that this field is both used
            as input and output. num_comps includes the solvent component. Note again that
            this function DO NOT initialize omegas, it should be initialized externally,
            and usually a random initialization will be a reasonable choice.
        Js (np.ndarray):
            initialization and output, the normalized volumes of the phases. 1D array with
            size of num_phases. The average value of Js will and should be unity. Note
            that this field is both used as input and output. A all-one array is usually a
            nice initialization, unless resume of a previous run is intended.
        phis (np.ndarray):
            output, the volume fractions. 2D array with size of num_comps-by-num_phases.
            num_comps includes the solvent component.
        steps_inner (int):
            hyperparameter, number of steps in current routine. Within these steps,
            convergence is not checked and no output will be generated.
        acceptance_Js (float):
            hyperparameter, The acceptance of Js. This value determines the amount of
            changes accepted in each step for the Js field. Typically this value can take
            the order of 10^-3, or smaller when the system becomes larger or stiffer.
        Js_step_upperbound (float):
            hyperparameter, The maximum change of Js per step. This values determines the
            maximum amount of changes accepted in each step for the Js field. If the
            intended amount is larger this value, the changes will be scaled down to
            guarantee that the maximum changes do not exceed this value. Typically this
            value can take the order of 10^-3, or smaller when the system becomes larger
            or stiffer.
        acceptance_omega (float):
            hyperparameter, The acceptance of omegas. This value determines the amount of
            changes accepted in each step for the omega field. Note that if the iteration
            of Js is scaled down due to parameter `Js_step_upperbound`, the iteration of
            omega fields will be scaled down simultaneously. Typically this value can take
            the order of 10^-2, or smaller when the system becomes larger or stiffer.
        kill_threshold (float):
            hyperparameter, The threshold of the Js for a phase to be killed. Should be
            not less than 0. In each iteration step, the Js array will be checked, for
            each element smaller than this parameter, the corresponding phase will be
            killed and 0 will be assigned to the corresponding mask. The dead phase may be
            revived, depending whether reviving is allowed or whether the `revive_tries`
            has been exhausted.
        revive_tries (int):
            hyperparameter, number of tries left to revive the dead phase. 0 or negative
            value indicates no reviving. WHen this value is exhausted, i.e. the number of
            revive in current function call exceeds this value, the revive will be turned
            off. Note that this function do not decrease this value, but return the number
            of revive after completion.
        revive_scaler (float):
            hyperparameter, the factor for the conjugate fields when a dead phase is
            revived. This value determines the range of the random conjugate field
            generated by the algorithm. TYpically 1.0 or some value slightly larger will
            be a reasonable choice.
        rng (np.random.Generator):
            random number generator, random number generator for reviving.

    Returns:
        Tuple[float, float, float, int, bool]: max incompressibility, max omega error, max
        J error, number of revive, whether no phase is killed in the last step
    """
    num_comps, num_phases = omegas.shape
    chi_sum_sum = chis.sum()

    n_valid_phase = 0

    for _ in range(steps_inner):
        revive_count = 0
        # check if there is invalid phase.
        if revive_count < revive_tries:
            n_valid_phase = np.sum(Js > kill_threshold)
            if n_valid_phase != num_phases:
                # kill and revive the phase

                # obtain the range for the rng
                omega_centers = np.full(num_comps, 0.0, float)
                omega_widths = np.full(num_comps, 0.0, float)
                for itr_comp in range(num_comps):
                    current_omega_min = omegas[itr_comp].min()
                    current_omega_max = omegas[itr_comp].max()
                    omega_centers[itr_comp] = (
                        current_omega_max + current_omega_min
                    ) * 0.5
                    omega_widths[itr_comp] = (
                        current_omega_max - current_omega_min
                    ) * 0.5

                # revive the phase with random conjugate field
                for itr_phase in range(num_phases):
                    if Js[itr_phase] <= kill_threshold:
                        Js[itr_phase] = 1.0
                        revive_count += 1
                        for itr_comp in range(num_comps):
                            omegas[itr_comp, itr_phase] = omega_centers[
                                itr_comp
                            ] + omega_widths[itr_comp] * revive_scaler * rng.uniform(
                                -1, 1
                            )

        # generate masks for the compartments
        mask = np.sign(Js - kill_threshold).clip(0.0)
        n_valid_phase = int(mask.sum())
        Js *= mask

        # calculate single molecular partition function Q
        Qs = np.full(num_comps, 0.0, float)
        for itr_comp in range(num_comps):
            Qs[itr_comp] = (np.exp(-omegas[itr_comp] * sizes[itr_comp]) * Js).sum()
            Qs[itr_comp] /= n_valid_phase

        # volume fractions and incompressibility
        incomp = np.full(num_phases, -1.0, float)
        for itr_comp in range(num_comps):
            factor = phi_means[itr_comp] / Qs[itr_comp]
            phis[itr_comp] = factor * np.exp(-omegas[itr_comp] * sizes[itr_comp]) * mask
            incomp += phis[itr_comp]
        incomp *= mask
        max_abs_incomp = np.abs(incomp).max()

        # temp for omega, namely chi.phi
        omega_temp = chis @ phis

        # xi, the lagrangian multiplier
        xi = chi_sum_sum * incomp
        for itr_comp in range(num_comps):
            xi += omegas[itr_comp] - omega_temp[itr_comp]
        xi *= mask
        xi /= num_comps

        # local energy. i.e. energy of phases excluding the partition function part
        local_energy = (xi + 1.0) * incomp + 1.0
        for itr_comp in range(num_comps):
            local_energy += (-0.5 * omega_temp[itr_comp] - xi) * phis[itr_comp]

        # Js is updated according to this local energy
        local_energy_mean = (local_energy * Js).sum() / n_valid_phase
        Js_diff = (local_energy_mean - local_energy) * mask
        max_abs_js_diff = np.abs(Js_diff).max()
        Js_max_change = max_abs_js_diff * acceptance_Js
        additional_factor = Js_step_upperbound / max(Js_max_change, Js_step_upperbound)
        Js += additional_factor * acceptance_Js * Js_diff
        Js *= mask
        Js += 1 - (Js.sum() / n_valid_phase)
        Js *= mask

        # update omega
        max_abs_omega_diff = 0
        for itr_comp in range(num_comps):
            omega_temp[itr_comp] = omega_temp[itr_comp] + xi - omegas[itr_comp]
            omega_temp[itr_comp] *= mask
            omega_temp[itr_comp] -= omega_temp[itr_comp].sum() / n_valid_phase
            max_abs_omega_diff = max(max_abs_omega_diff, omega_temp[itr_comp].max())
            omegas[itr_comp] += (
                additional_factor * acceptance_omega * omega_temp[itr_comp]
            )
            omegas[itr_comp] *= mask
            omegas[itr_comp] -= omegas[itr_comp].sum() / n_valid_phase

    # count the valid phases in the last step
    n_valid_phase_last = 0
    for J in Js:
        if J > kill_threshold:
            n_valid_phase_last += 1

    return (
        max_abs_incomp,
        max_abs_omega_diff,
        max_abs_js_diff,
        revive_count,
        n_valid_phase == n_valid_phase_last,
    )


def make_fixed_Js_and_phis_by_duplication(
    Js: np.ndarray,
    phis: np.ndarray,
    *,
    kill_threshold: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    replace the dead phases by the duplication of living phases, without
    changing the solution. Note that this function should only be used for
    records and compatibility, and should not be treated as a way to revive a
    dead phase. Therefore, the omega field is not modified by this function.

    Args:
        Js (np.ndarray):
            input, the normalized volumes of the phases. 1D array with size of
            num_phases. The masked mean value, i.e. Js[Js>kill_threshold].mean()
            , of this array should be one. Note that this is not checked by the
            function.
        phis (np.ndarray):
            output, the volume fractions. 2D array with size of
            num_comps-by-num_phases. Note that the incompressibility is not
            checked by the function.
        kill_threshold (float):
            parameter, The threshold of the Js for a phase to be killed. Js
            array will be checked, for each element smaller than this parameter,
            the corresponding phase will be considered as dead, and its phis
            values will then be copied from a random living one, while its Js
            value will set to half of that of the living one. In the end, the
            returned Js will be normalized.
        rng (np.random.Generator):
            random number generator, random number generator for reviving.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Js_fixed, phis_fixed
    """
    _, num_phases = phis.shape

    Js_fixed = Js.copy("C")
    phis_fixed = phis.copy("C")

    dead_indexes = []
    living_nicely_indexes = []
    for itr_phases in range(num_phases):
        if Js_fixed[itr_phases] > 2.0 * kill_threshold:
            living_nicely_indexes.append(itr_phases)
        elif Js_fixed[itr_phases] <= kill_threshold:
            dead_indexes.append(itr_phases)

    for itr_dead in dead_indexes:
        while True:
            pos_in_living = rng.integers(0, len(living_nicely_indexes))
            ref_index = living_nicely_indexes[pos_in_living]
            if Js_fixed[ref_index] > 2.0 * kill_threshold:
                phis_fixed[:, itr_dead] = phis_fixed[:, ref_index]
                Js_fixed[itr_dead] = 0.5 * Js_fixed[ref_index]
                Js_fixed[ref_index] = 0.5 * Js_fixed[ref_index]
                living_nicely_indexes.append(itr_dead)
                break
            else:
                living_nicely_indexes.pop(pos_in_living)

    Js_fixed /= Js_fixed.mean()
    return Js_fixed, phis_fixed


class CoexistingPhasesFinder:
    def __init__(
        self,
        chis: np.ndarray,
        phi_means: np.ndarray,
        num_compartment: int,
        *,
        sizes: np.ndarray = None,
        rng: np.random.Generator = None,
        convergence_criterion: str = "standard",
        random_std: float = 5.0,
        acceptance_Js: float = 0.0002,
        acceptance_omega: float = 0.002,
        Js_step_upperbound: float = 0.001,
        kill_threshold: float = 0.0,
        revive_scaler: float = 1.0,
        revive_count_multiplier: int = 16,
        additional_chi_shift: float = 1.0,
    ):
        self._logger = logging.getLogger(self.__class__.__name__)

        # chis
        if chis.shape[0] == chis.shape[1]:
            self.chis = chis
            self.num_component = chis.shape[0]
            self._logger.info(
                f"We infer that there are {self.num_component} components in the system from the chis matrix."
            )
        else:
            self._logger.error(f"chis matrix with size of {chis.shape} is not square.")
            raise ValueError("chis matrix must be square.")

        # phi_means
        if phi_means.shape[0] == self.num_component:
            self.phi_means = phi_means
            if np.abs(self.phi_means.sum() - 1.0) > 1e-12:
                self._logger.warn(
                    f"Total phi_means is not 1.0. Iteration may never converge."
                )
        else:
            self._logger.error(
                f"phi_means vector with size of {phi_means.shape} is invalid, since {self.num_component} is defined by chis matrix."
            )
            raise ValueError(
                "phi_means vector must imply same component number as chis matrix."
            )

        self.num_compartment = num_compartment

        ## optional arguments

        # sizes
        if sizes is None:
            self.sizes = np.ones(self.num_component)
        else:
            if sizes.shape[0] == self.num_component:
                self.sizes = sizes
                if np.sum(self.sizes <= 0):
                    self._logger.warn(
                        f"Non-positive sizes detected. Iteration will probably fail."
                    )
            else:
                self._logger.error(
                    f"sizes vector with size of {sizes.shape} is invalid, since {self.num_component} is defined by chis matrix."
                )
                raise ValueError(
                    "sizes vector must imply same component number as chis matrix."
                )

        # rng
        if rng is None:
            self.rng_is_external = False
            self.rng_seed = int(datetime.now().timestamp())
            self.rng = np.random.default_rng(self.rng_seed)
        else:
            self.rng_is_external = True
            self.rng_seed = int(0)
            self.rng = rng

        # other parameters
        self.convergence_criterion = convergence_criterion
        self.random_std = random_std
        self.acceptance_Js = acceptance_Js
        self.acceptance_omega = acceptance_omega
        self.Js_step_upperbound = Js_step_upperbound
        self.kill_threshold = kill_threshold
        self.revive_scaler = revive_scaler
        self.revive_count_multiplier = revive_count_multiplier
        self.additional_chi_shift = additional_chi_shift

        ## initialize derived internal states
        self.Js = np.full(self.num_compartment, 0.0, float)
        self.omegas = np.full((self.num_component, self.num_compartment), 0.0, float)
        self.revive_count_left = self.revive_count_multiplier * self.num_compartment
        self.reinitialize_random()

    def reinitialize_random(self):
        self.omegas = self.rng.normal(
            0.0,
            self.random_std,
            (self.num_component, self.num_compartment),
        )
        self.Js = np.full(self.num_compartment, 1.0, float)
        self.revive_count_left = self.revive_count_multiplier * self.num_compartment

    def reinitialize_from_omegas(self, omegas):
        if omegas.shape == self.omegas.shape:
            self.omegas = omegas
        else:
            self._logger.error(
                f"new omegas with size of {omegas.shape} is invalid. It must have the size of {(self.num_component, self.num_compartment)}."
            )
            raise ValueError("New omegas must match the size of the old one.")
        self.Js = np.ones_like(self.Js)
        self.revive_count_left = self.revive_count_multiplier * self.num_compartment

    def reinitialize_from_phis(self, phis):
        if phis.shape == self.omegas.shape:
            self.omegas = -np.log(phis)
            for itr in range(self.num_component):
                self.omegas[itr] /= self.sizes[itr]
        else:
            self._logger.error(
                f"phis with size of {phis.shape} is invalid. It must have the size of {(self.num_component, self.num_compartment)}."
            )
            raise ValueError("phis must match the size of the omegas.")
        self.Js = np.ones_like(self.Js)
        self.revive_count_left = self.revive_count_multiplier * self.num_compartment

    def set_phi_means(self, phi_means):
        if phi_means.shape == self.phi_means.shape:
            self.phi_means = phi_means
        else:
            self._logger.error(
                f"new phi_means with size of {phi_means.shape} is invalid. It must have the size of {(self.num_component)}."
            )
            raise ValueError("New phi_means must match the size of the old one.")

    def evolve(
        self,
        t_range: float,
        *,
        interval: float = 1000.0,
        tolerance: float = 1e-5,
        progress: bool = True,
        save_intermediate_data: bool = False,
    ):
        """use explicit Euler method with an adaptive time step to evolve ODEs

        Args:
            t_range:
                The time for which the simulation is maximally run. It might abort earlier
                when the stationary state is reached. Since in class
                :class:`FieldLikeRelaxationDynamics` there is no unique definition of time
                step, this value will be casted to int and treated as number of steps.
            interval:
                The time interval at which convergence is checked (and data is saved if
                `save_intermediate_data` is True. Since in class
                :class:`FieldLikeRelaxationDynamics` there is no unique definition of time
                step, this value will be casted to int and treated as number of steps.
            tolerance:
                The tolerance determining when the simulation will be aborted because the
                algorithm thinks we reached the stationary state
            progress:
                Flag determining whether to show a progress bar during the simulation.
            save_intermediate_data:
                Flag determining whether saving (and returning) data in time intervals
                determined by `interval`

        Returns:
            tuple: Simulation time and mixture. If `save_intermediate_data == True`, these
            are lists with all the saved data. Otherwise, only the last time point is
            returned.
        """
        steps_tracker = int(np.ceil(t_range / interval))
        steps_inner = max(1, int(np.ceil(t_range)) // steps_tracker)

        num_steps = 0
        t = 0.0

        chis_shifted = self.chis.copy()
        chis_shifted += -self.chis.min() + self.additional_chi_shift
        phis = np.zeros_like(self.omegas)

        mixtures = []
        times = []

        pbar = tqdm(range(steps_tracker), disable=not progress)
        for _ in pbar:
            # store result
            if save_intermediate_data:
                times.append(t)
                (
                    volumes_intermediate,
                    phis_intermediate,
                ) = make_fixed_Js_and_phis_by_duplication(
                    self.Js,
                    phis,
                    kill_threshold=self.kill_threshold,
                    rng=self.rng,
                )
                volumes_intermediate /= volumes_intermediate.sum()
                mixtures.append((phis_intermediate, volumes_intermediate))

            # do the inner steps
            (
                max_abs_incomp,
                max_abs_omega_diff,
                max_abs_js_diff,
                revive_count,
                is_last_step_safe,
            ) = find_coexisting_phases_impl(
                self.phi_means,
                self.chis,
                self.sizes,
                omegas=self.omegas,
                Js=self.Js,
                phis=phis,
                steps_inner=steps_inner,
                acceptance_Js=self.acceptance_Js,
                Js_step_upperbound=self.Js_step_upperbound,
                acceptance_omega=self.acceptance_omega,
                kill_threshold=self.kill_threshold,
                revive_tries=self.revive_count_left,
                revive_scaler=self.revive_scaler,
                rng=self.rng,
            )

            pbar.set_description(
                "max_abs_incomp={:6.4f} max_abs_omega_diff={:6.4f} max_abs_js_diff={:6.4f}".format(
                    max_abs_incomp, max_abs_omega_diff, max_abs_js_diff
                )
            )

            num_steps += steps_inner
            t += steps_inner

            self.revive_count_left -= revive_count

            # check convergence
            if self.convergence_criterion == "standard":
                if (
                    is_last_step_safe
                    and tolerance > max_abs_incomp
                    and tolerance > max_abs_omega_diff
                    and tolerance > max_abs_js_diff
                ):
                    self._logger.info("Composition and sizes reached stationary state")
                    break
            else:
                raise ValueError(
                    f"Undefined convergence criterion: {self.convergence_criterion}"
                )

        # add final data point
        if not times or times[-1] != t:
            times.append(t)
            (
                volumes_intermediate,
                phis_intermediate,
            ) = make_fixed_Js_and_phis_by_duplication(
                self.Js,
                phis,
                kill_threshold=self.kill_threshold,
                rng=self.rng,
            )
            volumes_intermediate /= volumes_intermediate.sum()
            mixtures.append((phis, volumes_intermediate))

        # store diagnostic output
        self.diagnostics = {
            "num_steps": num_steps,
            "steps_tracker": steps_tracker,
        }

        if save_intermediate_data:
            return np.array(times), mixtures
        else:
            return times[0], mixtures[0]
