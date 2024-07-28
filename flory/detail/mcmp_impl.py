""" The implementation details of the module :mod:`~flory.mcmp`. 

:mod:`flory.detail.mcmp_impl` contains the implementation details of the module
:mod:`~flory.mcmp`. The main components of the module is the function
:meth:`multicomponent_self_consistent_metastep`, which implements the self consistent
iterations for minimizing the extended free energy functional, and the function
:meth:`get_clusters`, which finds the unique phases.

In this module, arguments of functions are always marked by `Const`, `Output` or
`Mutable`, to indicate whether the arguments will keep invariant, be directly overwritten,
or reused.

.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numba as nb
import numpy as np
from scipy import cluster, spatial


@nb.njit()
def count_valid_compartments(Js: np.ndarray, threshold: float) -> int:
    """
    Count how many entries in :paramref:`~count_valid_compartments.Js` are larger than
    :paramref:`~count_valid_compartments.threshold`.

    Args:
        Js:
            Const. 1D array with the size of :math:`M`.
        threshold:
            Const. The threshold value.

    Returns:
        : Number of entries in :paramref:`Js` larger than :paramref:`threshold`.
    """
    return (Js > threshold).sum()


@nb.njit()
def make_valid_compartment_masks(Js: np.ndarray, threshold: float) -> np.ndarray:
    """
    Create masks for entries in :paramref:`~make_valid_compartment_masks.Js` are larger
    than :paramref:`~make_valid_compartment_masks.threshold`. Value of 1.0 or 0.0
    indicates a valid or invalid mask, respectively.

    Args:
        Js:
            Const. 1D array with the size of :math:`M`.
        threshold:
            Const. The threshold value.

    Returns:
        : 
            1D array with the size of :math:`M`, containing masks of entries in
            :paramref:`Js` larger than :paramref:`threshold`.
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
    Randomly revive compartments whose relative volume (element of
    :paramref:`~revive_compartments_by_random.Js`) is smaller than
    :paramref:`~revive_compartments_by_random.threshold`. The revived values are randomly
    and uniformly sampled between the extreme values of :paramref:`targets` across all
    compartments. The range can be scaled by the parameter :paramref:`scaler`. Note that
    this function does not conserve the quantities in :paramref:`targets` across all
    compartments, since the new values are randomly generated.

    Args:
        Js:
            Mutable. 1D array with the size of :math:`M`, containing values to determine
            whether a compartment is dead.
        targets:
            Mutable. 2D array with the size of :math:`N_\\mathrm{c} \\times M`, containing
            the values to be revived. The second dimension has to be the same as that of
            Js. Note that this is not checked.
        threshold:
            Const. The threshold of :paramref:`Js` for a compartment to be considered
            dead. For each element of :paramref:`Js` smaller than this parameter, the
            corresponding compartment will be considered as dead, and its targets values
            will then be randomly drawn between the corresponding minimum and maximum
            values of :paramref:`targets` across all compartments. Corresponding
            :paramref:`Js` will be set to be unity.
        rng:
            Mutable. Random number generator for reviving.
        scaler:
            Const. The scaler for generating random new values.

    Returns:
        : Number of dead compartments that have been revived.
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

    # revive the compartment with random conjugate field
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
    Revive compartments whose relative volume (element of
    :paramref:`~revive_compartments_by_copy.Js`) is smaller than
    :paramref:`~revive_compartments_by_copy.threshold`. The revived values are randomly
    copied from other living compartments. Note that this function conserves the
    quantities in :paramref:`targets` across all compartments by modifying the volumes
    :paramref:`Js` accordingly.

    Args:
        Js:
            Mutable. 1D array with the size of :math:`M`, containing values to determine
            whether a compartment is dead.
        targets:
            Mutable. 2D array with the size of :math:`N_\\mathrm{c} \\times M`, containing the
            values to be revived. The second dimension has to be the same as that of Js.
            Note that this is not checked.
        threshold:
            Const. The threshold of :paramref:`Js` for a compartment to be considered
            dead. For each element of :paramref:`Js` smaller than this parameter, the
            corresponding compartment will be considered as dead, and its targets values
            will then be randomly drawn between the corresponding minimum and maximum
            values of :paramref:`targets` across all compartments. Corresponding
            :paramref:`Js` will be set to be unity.
        rng:
            Mutable. Random number generator for reviving.

    Returns:
        : Number of revives
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
    Calculate the volume fractions of components :math:`\\phi_i^{(m)}` from the conjugate
    fields :math:`\\omega_i^{(m)}`. The single molecular partition functions $Q_i$ and the
    incompressibility :math:`\\sum_i \\phi_i^{(m)} - 1` are also calculated. Note that the
    volume fractions :math:`\\phi_i^{(m)}` are returned through the parameter.

    Args:
        phis:
            Output. The 2D array with the size of :math:`N_\\mathrm{c} \\times M`,
            containing the volume fractions :math:`\\phi_i^{(m)}`. The first dimension has
            to be the same as :paramref:`sizes`. The second dimension has to be the same
            as that of :paramref:`Js`. Note that these are not checked.
        Js:
            Const. The 1D array with the size of :math:`M`, containing the relative
            volumes of compartments :math:`J_m`. Note that :paramref:`Js` must be
            invariant under multiplication of :paramref:`masks`.
        sizes:
            Const. The 1D array with the size of :math:`M`, containing the relative
            molecule volumes of the components :math:`l_i`.
        phi_means:
            Const. The 1D array with the size of :math:`N_\\mathrm{c}`, containing the
            mean volume fractions of the components :math:`\\bar{\\phi}_i`.
        omegas:
            Const. The 2D array with the size of :math:`N_\\mathrm{c} \\times M`,
            containing the conjugate field :math:`\\omega_i^{(m)}`. The first dimension
            has to be the same as :paramref:`sizes`. The second dimension has to be the
            same as that of :paramref:`Js`. Note that these are not checked.
        masks:
            Const. The 1D array with the size of :math:`M`, containing the masks of
            compartments. See :meth:`make_valid_compartment_masks` for more information.

    Returns:
        [0]: 
            1D array with the size of :math:`N_\\mathrm{c}`, containing the single
            molecular partition functions of components :math:`Q_i`. 
        [1]: 
            1D array with the size of :math:`M`, containing the incompressibility
            :math:`\\sum_i \\phi_i^{(m)} - 1`.
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
    The core algorithm of finding coexisting states of multicomponent systems with
    self-consistent iterations.

    Args:
        phi_means:
            Const. The interaction matrix. 2D array with size of :math:`N_\\mathrm{c}
            \\times N_\\mathrm{c}`. This matrix should be the full :math:`\\chi_{ij}`
            matrix of the system, including the solvent component. Note that the matrix
            must be symmetric, which is not checked but should be guaranteed externally.
        chis:
            Const. The average volume fractions :math:`\\bar{\\phi}_i` of all the
            components of the system. 1D array with size of :math:`N_\\mathrm{c}`. Note
            that the volume fraction of the solvent is included as well, therefore the sum
            of this array must be unity, which is not checked by this function and should
            be guaranteed externally.
        sizes:
            Const. The relative molecule volumes :math:`l_i` of the components. 1D array
            with size of :math:`N_\\mathrm{c}`. This sizes vector should be the full sizes
            vector of the system, including the solvent component. An element of one
            indicates that the corresponding specie has the same volume as the reference.
            None indicates a all-one vector.
        omegas:
            Mutable. The conjugate field :math:`\\omega_i^{(m)}`. 2D array with size of
            :math:`N_\\mathrm{c} \\times M`. Note that this field is both used as input
            and output. Note again that this function DO NOT initialize `omegas`, it
            should be initialized externally, and usually a random initialization will be
            a reasonable choice.
        Js:
            Mutable. The relative volumes of the compartments :math:`J_m`. 1D array with
            size of :math:`M`. The average value of `Js` will and should be unity. Note
            that this field is both used as input and output. An all-one array is usually
            a nice initialization, unless resume of a previous run is intended.
        phis:
            Output. The volume fractions :math:`\\phi_i^{(m)}`. 2D array with size of
            :math:`N_\\mathrm{c} \\times M`.
        steps_inner:
            Constant. Number of steps in current routine. Within these steps, convergence
            is not checked and no output will be generated.
        acceptance_Js:
            Constant. The acceptance of :paramref:`Js` (the relative compartment size
            :math:`J_m`). This value determines the amount of changes accepted in each
            step for the :math:`J_m` field. Typically this value can take the order of
            :math:`10^{-3}`, or smaller when the system becomes larger or stiffer.
        Js_step_upper_bound:
            Constant. The maximum change of :paramref:`Js` (the relative compartment size
            :math:`J_m`) per step. This value is designed to reduce the risk that a the
            volume of a compartment changes too fast before it develops meaningful
            composition. If the intended change is larger this value, all the changes will
            be scaled down to guarantee that the maximum changes do not exceed this value.
            Typically this value can take the order of :math:`10^{-3}`, or smaller when
            the system becomes larger or stiffer. 
        acceptance_omega:
            Constant. The acceptance of :paramref:`omegas`(the conjugate fields
            :math:`\\omega_i^{(m)}`). This value determines the amount of changes accepted
            in each step for the :math:`\\omega_i^{(m)}` field. Note that if the iteration
            of :math:`J_m` is scaled down due to parameter
            :paramref:`Js_step_upper_bound`, the iteration of :math:`\\omega_i^{(m)}`
            fields will be scaled down simultaneously. Typically this value can take the
            order of :math:`10^{-2}`, or smaller when the system becomes larger or
            stiffer.         
        kill_threshold:
            Constant. The threshold of the :math:`J_m` for a compartment to be considered
            dead and killed afterwards. Should be not less than 0. In each iteration step,
            the :math:`J_m` array will be checked, for each element smaller than this
            parameter, the corresponding compartment will be killed and 0 will be assigned
            to the internal mask. The dead compartment may be revived, depending whether
            reviving is allowed or whether the number of the revive tries has been
            exhausted. 
        revive_tries:
            Constant. Number of tries left to revive the dead compartment. 0 or negative
            value indicates no reviving. When this value is exhausted, i.e. the number of
            revive in current function call exceeds this value, the revive will be turned
            off. Note that this function does not decrease this value, but returns the number
            of revives that have happened after completion.
        revive_scaler:
            Constant. The scaling factor for the conjugate fields :math:`\\omega_i^{(m)}`
            when a dead compartment is revived. This value determines the range of the
            random conjugate field generated by the algorithm. Typically 1.0 or some value
            slightly larger will be a reasonable choice. See
            :meth:`revive_compartments_by_random` for more information.
        rng:
            Mutable. random number generator for reviving.

    Returns:
        [0]: Max absolute incompressibility.
        [1]: Max absolute conjugate field error.
        [2]: Max absolute relative volumes error.
        [3]: Number of revives.
        [4]: Whether no phase is killed in the last step.
    """
    num_components, num_compartments = omegas.shape
    chi_sum_sum = chis.sum()

    n_valid_phase = 0

    revive_count = 0
    for _ in range(steps_inner):
        # check if we are still allowed to revive compartments
        if revive_count < revive_tries:
            n_valid_phase = count_valid_compartments(Js, kill_threshold)
            if n_valid_phase != num_compartments:
                # revive dead compartments
                revive_count += revive_compartments_by_random(
                    Js, omegas, kill_threshold, rng, revive_scaler
                )

        # generate masks for the compartments
        masks = make_valid_compartment_masks(Js, kill_threshold)
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
    n_valid_phase_last = count_valid_compartments(Js, kill_threshold)

    return (
        max_abs_incomp,
        max_abs_omega_diff,
        max_abs_Js_diff,
        revive_count,
        n_valid_phase == n_valid_phase_last,
    )


def sort_phases(
    Js_phases: np.ndarray, phis_phases: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sort the phases according to the index of most concentrated components. Note that this
    function uses different data structure from other functions in the module. See
    :paramref:`phis_phases`.

    Args:
        Js_phases:
            Const. 1D array with the size of :math:`N_\\mathrm{p}`, containing the volumes
            of each phase.
        phis_phases:
            Const. 2D array with the size of :math:`N_\\mathrm{p} \\times N_\\mathrm{c}`,
            containing the volume fractions of the components in each phase. The first
            dimension must be the same as :paramref:`Js_phases`. Note that this usually
            corresponds to the transpose of the arrays of :math:`\\phi_i^{(m)}`, for
            example :attr:`~flory.mcmp.CoexistingPhasesFinder.phis` in class
            :class:`~flory.mcmp.CoexistingPhasesFinder`.

    Returns:
        [0]: Sorted :paramref:`Js_phases`.
        [1]: Sorted :paramref:`phis_phases`.
    """
    enrich_indexes = np.argsort(phis_phases)
    sorting_index = np.lexsort(np.transpose(enrich_indexes))
    return Js_phases[sorting_index], phis_phases[sorting_index]


def get_clusters(
    Js: np.ndarray, phis: np.ndarray, dist: float = 1e-2
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find unique phases from compartments by clustering. The returning results are sorted
    according to the index of most concentrated components.

    Args:
        Js:
            Const. 1D array with the size of :math:`M`, containing the volumes of each
            compartment.
        phis:
            Const. 2D array with the size of :math:`N_\\mathrm{c} \\times M`, containing
            the volume fractions of the components in each phase :math:`\\phi_i^{(m)}`.
            The second dimension must be the same as :math:`Js`.
        dist: 
            Cut-off distance for cluster analysis
    
    Returns:
        [0]: 
            1D array with the size of :math:`N_\\mathrm{p}`, containing the volumes of the
            unique phases.
        [1]: 
            1D array with the size of :math:`N_\\mathrm{p} \\times  N_\\mathrm{c}`,
            containing the compositions of all unique phases. Note that the data structure
            is different from normal, see :meth:`sort_phases` for more information.
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
    cluster_Js = np.array([Js[clusters == n].sum(axis=0) for n in np.unique(clusters)])
    cluster_Js /= cluster_Js.sum()
    return sort_phases(cluster_Js, cluster_phis)
