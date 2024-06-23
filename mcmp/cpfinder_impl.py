"""
The implementations for finding coexisting phases 

.. autosummary::
   :nosignatures:

.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numba as nb
import numpy as np
from scipy import cluster, spatial


@nb.njit()
def count_valid_phases(Js: np.ndarray, threshold: float) -> int:
    """
    Count how many entries in `Js` are larger than `threshold`
    
    Args:
        Js (np.ndarray):
            Const. 1D array of values to check.
        threshold (float):
            Const.

    Returns:
        int: Number of entries in `Js` larger than `threshold`
    """
    return (Js > threshold).sum()


@nb.njit()
def make_valid_phase_masks(Js: np.ndarray, threshold: float) -> np.ndarray:
    """
    Create masks for entries in `Js` are larger than `threshold`
    
    Args:
        Js (np.ndarray):
            Const. 1D array of values to check.
        threshold (float):
            Const.

    Returns:
        np.ndarray: Masks of entries in `Js` larger than `threshold`
    """
    return np.sign(Js - threshold).clip(0.0)


@nb.njit()
def revive_compartments_by_random(
    Js: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    rng: np.random.Generator,
    scaler: float,
) -> int:
    """
    Revive compartment randomly whose `J` (element of `Js`) is smaller than threshold. The
    revived values are randomly and uniformly sampled between the min and max value of
    `targets` across all compartments, which can be scaled by the scaler.
    
    Args:
        Js (np.ndarray):
            Mutable. The 1D array values to determine whether a compartment is dead.
        targets (np.ndarray):
            Mutable. The 2D array of the values to be revived. The second dimension has to
            be the same as that of Js. Note that this is not checked.
        threshold (float):
            Const. The threshold of `Js` for a compartment to be considered dead. For each
            element of `Js` smaller than this parameter, the corresponding compartment
            will be considered as dead, and its targets values will then be randomly drawn
            between the corresponding min and max value of `targets` across all
            compartments. Corresponding `Js` will be set to be unity.
        rng (np.random.Generator):
            Mutable. Random number generator for reviving.
        scaler (float):
            Const. The scaler for generating new values.

    Returns:
        int: Number of revives
    """
    revive_count = 0
    num_component, num_compartment = targets.shape

    target_centers = np.full(num_component, 0.0, float)
    omega_widths = np.full(num_component, 0.0, float)
    for itr_component in range(num_component):
        current_target_max = targets[itr_component].max()
        current_target_min = targets[itr_component].min()
        target_centers[itr_component] = (current_target_max + current_target_min) * 0.5
        omega_widths[itr_component] = (current_target_max - current_target_min) * 0.5

    # revive the phase with random conjugate field
    for itr_compartment in range(num_compartment):
        if Js[itr_compartment] <= threshold:
            Js[itr_compartment] = 1.0
            revive_count += 1
            for itr_component in range(num_component):
                targets[itr_component, itr_compartment] = target_centers[
                    itr_component
                ] + omega_widths[itr_component] * scaler * rng.uniform(-1, 1)

    return revive_count


@nb.njit()
def revive_compartments_by_copy(
    Js: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    rng: np.random.Generator,
) -> int:
    """
    Revive compartment whose `J` (element of `Js`) is smaller than threshold. The revived
    values are randomly copied from other living compartments.

    Args:
        Js (np.ndarray):
            Mutable. The 1D array values to determine whether a compartment is dead.
        targets (np.ndarray):
            Mutable. The 2D array of the values to be revived. The second dimension has to
            be the same as that of Js. Note that this is not checked.
        threshold (float):
            Const. The threshold of `Js` for a compartment to be considered dead. For each
            element of `Js` smaller than this parameter, the corresponding compartment
            will be considered as dead, and its targets values will then be randomly
            copied from living compartments. Corresponding `Js` will be set to be unity.
        rng (np.random.Generator):
            Mutable. Random number generator for reviving.

    Returns:
        int: Number of revives
    """
    revive_count = 0
    num_components, num_compartments = targets.shape

    dead_indexes = np.full(num_compartments, -1, dtype=np.int32)
    dead_count = 0
    living_nicely_indexes = np.full(num_compartments, -1, dtype=np.int32)
    living_nicely_count = 0
    for itr_compartment in range(num_compartments):
        if Js[itr_compartment] > 2.0 * threshold:
            living_nicely_indexes[living_nicely_count] = itr_compartment
            living_nicely_count += 1
        elif Js[itr_compartment] <= threshold:
            dead_indexes[dead_count] = itr_compartment
            dead_count += 1

    for itr_dead in dead_indexes[:dead_count]:
        while True:
            pos_in_living = rng.integers(0, living_nicely_count)
            ref_index = living_nicely_indexes[pos_in_living]
            if Js[int(ref_index)] > 2.0 * threshold:
                targets[:, itr_dead] = targets[:, ref_index]
                new_J = 0.5 * Js[ref_index]
                Js[itr_dead] = new_J
                Js[ref_index] = new_J
                living_nicely_indexes[living_nicely_count] = itr_dead
                living_nicely_count += 1
                revive_count += 1
                break
            else:
                living_nicely_count -= 1
                living_nicely_indexes[pos_in_living] = living_nicely_indexes[
                    living_nicely_count
                ]
                living_nicely_indexes[living_nicely_count] = -1
    return revive_count


@nb.njit()
def calc_volume_fractions(
    phis: np.ndarray,
    Js: np.ndarray,
    sizes: np.ndarray,
    phi_means: np.ndarray,
    omegas: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the volume fractions of components. The single molecular partition functions
    and the incompressibility are also calculated. Note that the volume fractions are
    returned through the parameter.

    Args:
        phis (np.ndarray):
            Output. The 2D array of the volume fractions. The first dimension has to be
            the same as `sizes`. The second dimension has to be the same as that of `Js`.
            Note that these are not checked.
        Js (np.ndarray):
            Const. The 1D array of volumes of compartments. Note that `Js` must be
            compatible with `masks`, which means `Js == Js * masks`.
        sizes (np.ndarray):
            Const. The 1D array of sizes of the components.
        phi_means (np.ndarray):
            Const. The 1D array of the mean volume fractions of the components.
        omegas (np.ndarray):
            Const. The 2D array of the conjugate fields. The first dimension has to be the
            same as `sizes`. The second dimension has to be the same as that of `Js`. Note
            that these are not checked.
        masks (np.ndarray):
            Const. The 1D array of masks of compartments.

    Returns:
        np.ndarray: the single molecular partition functions of components np.ndarray: the
        incompressibility
    """

    num_components, num_compartments = omegas.shape
    Qs = np.full(num_components, 0.0, float)
    incomp = np.full(num_compartments, -1.0, float)
    total_Js = Js.sum()
    for itr_comp in range(num_components):
        phis[itr_comp] = np.exp(-omegas[itr_comp] * sizes[itr_comp])
        Qs[itr_comp] = (phis[itr_comp] * Js).sum()
        Qs[itr_comp] /= total_Js
        factor = phi_means[itr_comp] / Qs[itr_comp]
        phis[itr_comp] = factor * phis[itr_comp] * masks
        incomp += phis[itr_comp]
    incomp *= masks
    return Qs, incomp


@nb.njit()
def multicomponent_self_consistent_metastep(
    phi_means: np.ndarray,
    chis: np.ndarray,
    sizes: np.ndarray,
    *,
    omegas: np.ndarray,
    Js: np.ndarray,
    phis: np.ndarray,
    steps_inner: int,
    acceptance_Js: float,
    Js_step_upper_bound: float,
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
            Const. The average volume fraction of all the components of the system. 1D
            array with size of num_components. Note that the volume fraction of the
            solvent is included as well, therefore the sum of this array must be unity,
            which is not checked by this function and should be guaranteed externally.
        chis (np.ndarray):
            Const. The interaction matrix. 2D array with size of
            num_components-by-num_components. This `chis` matrix should be the full `chis`
            matrix of the system, including the solvent component. Note that the symmetry
            is not checked, which should be guaranteed externally.
        sizes (np.ndarray):
            Const. The relative molecule volumes of the components. 1D array with size of
            num_components. This `sizes` vector should be the full `sizes` vector of the
            system, including the solvent component.
        omegas (np.ndarray):
            Mutable. The conjugate fields of the volume fractions. 2D array with size of
            num_components-by-num_compartments. Note that this field is both used as input
            and output. num_components includes the solvent component. Note again that
            this function DO NOT initialize `omegas`, it should be initialized externally,
            and usually a random initialization will be a reasonable choice.
        Js (np.ndarray):
            Mutable. The normalized volumes of the compartments. 1D array with size of
            num_compartments. The average value of `Js` will and should be unity. Note
            that this field is both used as input and output. An all-one array is usually
            a nice initialization, unless resume of a previous run is intended.
        phis (np.ndarray):
            Output. The volume fractions. 2D array with size of
            num_components-by-num_compartments. num_components includes the solvent
            component.
        steps_inner (int):
            Constant. Number of steps in current routine. Within these steps, convergence
            is not checked and no output will be generated.
        acceptance_Js (float):
            Constant. The acceptance of `Js`. This value determines the amount of changes
            accepted in each step for the `Js` field. Typically this value can take the
            order of 10^-3, or smaller when the system becomes larger or stiffer.
        Js_step_upper_bound (float):
            Constant. The maximum change of `Js` per step. This values determines the
            maximum amount of changes accepted in each step for the `Js` field. If the
            intended amount is larger this value, the changes will be scaled down to
            guarantee that the maximum changes do not exceed this value. Typically this
            value can take the order of 10^-3, or smaller when the system becomes larger
            or stiffer.
        acceptance_omega (float):
            Constant. The acceptance of `omegas`. This value determines the amount of
            changes accepted in each step for the `omegas` field. Note that if the
            iteration of `Js` is scaled down due to parameter `Js_step_upper_bound`, the
            iteration of `omegas` field will be scaled down simultaneously. Typically this
            value can take the order of 10^-2, or smaller when the system becomes larger
            or stiffer.
        kill_threshold (float):
            Constant. The threshold of the `Js` for a compartment to be killed. The value
            should be not less than 0. In each iteration step, the `Js` array will be
            checked. For each element smaller than this parameter, the corresponding
            compartment will be killed and 0 will be assigned to the corresponding mask.
            The dead compartment may be revived, depending whether reviving is allowed or
            whether the `revive_tries` has been exhausted.
        revive_tries (int):
            Constant. Number of tries left to revive the dead phase. 0 or negative value
            indicates no reviving. WHen this value is exhausted, i.e. the number of revive
            in current function call exceeds this value, the revive will be turned off.
            Note that this function do not decrease this value, but return the number of
            revive after completion.
        revive_scaler (float):
            Constant. The scaling factor for the conjugate fields when a dead phase is
            revived. This value determines the range of the random conjugate field
            generated by the algorithm. Typically 1.0 or some value slightly larger will
            be a reasonable choice.
        rng (np.random.Generator):
            Mutable. random number generator for reviving.

    Returns:
        Tuple[float, float, float, int, bool]: max incompressibility, max omega error, max
        J error, number of revive, whether no phase is killed in the last step
    """
    num_components, num_compartments = omegas.shape
    chi_sum_sum = chis.sum()

    n_valid_phase = 0

    for _ in range(steps_inner):
        revive_count = 0
        # check if we are still allowed to revive compartments
        if revive_count < revive_tries:
            n_valid_phase = count_valid_phases(Js, kill_threshold)
            if n_valid_phase != num_compartments:
                # revive dead compartments
                revive_count += revive_compartments_by_random(
                    Js, omegas, kill_threshold, rng, revive_scaler
                )

        # generate masks for the compartments
        masks = make_valid_phase_masks(Js, kill_threshold)
        n_valid_phase = int(masks.sum())
        Js *= masks

        # calculate volume fractions, single molecular partition function Q and incompressibility
        (Qs, incomp) = calc_volume_fractions(phis, Js, sizes, phi_means, omegas, masks)
        max_abs_incomp = np.abs(incomp).max()

        # temp for omega, namely chi.phi
        omega_temp = chis @ phis

        # xi, the lagrangian multiplier
        xi = chi_sum_sum * incomp
        for itr_comp in range(num_components):
            xi += omegas[itr_comp] - omega_temp[itr_comp]
        xi *= masks
        xi /= num_components

        # local energy. i.e. energy of phases excluding the partition function part
        local_energy = xi * incomp
        for itr_comp in range(num_components):
            local_energy += (
                -0.5 * omega_temp[itr_comp] - xi - 1.0 / sizes[itr_comp]
            ) * phis[itr_comp]

        # calculate the difference of Js
        local_energy_mean = (local_energy * Js).sum() / n_valid_phase
        Js_diff = (local_energy_mean - local_energy) * masks
        max_abs_Js_diff = np.abs(Js_diff).max()

        # calculate additional factor to scale down iteration
        Js_max_change = max_abs_Js_diff * acceptance_Js
        additional_factor = Js_step_upper_bound / max(Js_max_change, Js_step_upper_bound)

        # update Js
        Js += additional_factor * acceptance_Js * Js_diff
        Js *= masks
        Js += 1 - (Js.sum() / n_valid_phase)
        Js *= masks

        # calculate difference of omega and update omega directly
        max_abs_omega_diff = 0
        for itr_comp in range(num_components):
            omega_temp[itr_comp] = omega_temp[itr_comp] + xi - omegas[itr_comp]
            omega_temp[itr_comp] *= masks
            omega_temp[itr_comp] -= omega_temp[itr_comp].sum() / n_valid_phase
            max_abs_omega_diff = max(max_abs_omega_diff, omega_temp[itr_comp].max())
            omegas[itr_comp] += (
                additional_factor * acceptance_omega * omega_temp[itr_comp]
            )
            omegas[itr_comp] *= masks
            omegas[itr_comp] -= omegas[itr_comp].sum() / n_valid_phase

    # count the valid phases in the last step
    n_valid_phase_last = count_valid_phases(Js, kill_threshold)

    return (
        max_abs_incomp,
        max_abs_omega_diff,
        max_abs_Js_diff,
        revive_count,
        n_valid_phase == n_valid_phase_last,
    )


def sort_phases(Js_phases:np.ndarray, phis_phases:np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """
    Sort the phases according to the index of most concentrated components. 

    Args:
        Js_phases (np.ndarray):
            Const. 1D array of the volumes of each phase.
        phis_phases (np.ndarray):
            Const. 2D array containing the volume fractions of the components in each
            phase. The first dimension must be the same as `Js_phases`. Note that usually
            this corresponds the transpose of `phis` in class `CPFinder`. 

    Returns:
        np.ndarray: sorted `Js_phases`
        np.ndarray: sorted `phis_phases`
    """
    enrich_indexes = np.argsort(phis_phases)
    sorting_index = np.lexsort(np.transpose(enrich_indexes))
    return Js_phases[sorting_index], phis_phases[sorting_index]

def get_clusters(Js: np.ndarray, phis: np.ndarray, dist: float = 1e-2) -> tuple[int, np.ndarray,np.ndarray]:
    """
    Return the concentrations in the distinct clusters

    Args:
        Js (np.ndarray):
            Const. 1D array of the volumes of each phase.
        phis (np.ndarray):
            Const. 2D array containing the volume fractions of the components in each
            phase. The second dimension must be the same as `Js`. 
        dist (float): Cut-off distance for cluster analysis
    """
    # transpose to make the compartment index the first index
    phis = np.transpose(phis)
    # calculate distances between compositions
    if phis.shape[0] == 1:
        return phis, 1
    dists = spatial.distance.pdist(phis)
    # obtain hierarchy structure
    links = cluster.hierarchy.linkage(dists, method="centroid")
    # flatten the hierarchy by clustering
    clusters = cluster.hierarchy.fcluster(links, dist, criterion="distance")
    cluster_phis = np.array(
        [phis[clusters == n, :].mean(axis=0) for n in np.unique(clusters)]
    )
    cluster_Js = np.array(
        [Js[clusters == n].sum(axis=0) for n in np.unique(clusters)]
    )
    cluster_Js /= cluster_Js.sum()
    return sort_phases(cluster_Js, cluster_phis)
