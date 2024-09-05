"""The implementation details of the core algorithm for finder. 

:mod:`~flory.mcmp._finder_impl` contains the implementation details of the module
:mod:`~flory.mcmp.finder`. The main components of the module is the function
:func:`multicomponent_self_consistent_metastep`, which implements the self consistent
iterations for minimizing the extended free energy functional, and the function
:func:`get_clusters`, which finds the unique phases.

In this module, arguments of functions are always marked by `Constant`, `Output` or
`Mutable`, to indicate whether the arguments will be kept invariant, directly overwritten,
or reused.

.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""
from __future__ import annotations

import numba as nb
import numpy as np
from numba import literal_unroll

from ..constraint.base import ConstraintBaseCompiled
from ..ensemble.base import EnsembleBaseCompiled
from ..entropy.base import EntropyBaseCompiled
from ..interaction.base import InteractionBaseCompiled


@nb.njit()
def count_valid_compartments(Js: np.ndarray, threshold: float) -> int:
    r"""Count valid compartments.

    Count how many entries in :paramref:`Js` are larger than :paramref:`threshold`.

    Args:
        Js:
            Constant. The 1D array with the size of :math:`N_\mathrm{M}`, containing the relative
            volumes of compartments :math:`J_m`.
        threshold:
            Constant. The threshold below which the corresponding compartment is
            considered dead.

    Returns:
        : Number of entries in :paramref:`Js` larger than :paramref:`threshold`.
    """
    return (Js > threshold).sum()


@nb.njit()
def make_valid_compartment_masks(Js: np.ndarray, threshold: float) -> np.ndarray:
    r"""Create masks for valid compartments.

    Create masks for entries in :paramref:`Js` are larger than :paramref:`threshold`.
    Value of 1.0 or 0.0 indicates a valid or invalid mask, respectively.

    Args:
        Js:
            Constant. The 1D array with the size of :math:`N_\mathrm{M}`, containing the relative
            volumes of compartments :math:`J_m`.
        threshold:
            Constant. The threshold below which the corresponding compartment is
            considered dead.

    Returns:
        :
            1D array with the size of :math:`N_\mathrm{M}`, containing masks of entries in
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
    r"""Revive dead compartments randomly.

    Randomly revive compartments whose relative volume (element of :paramref:`Js`) is
    smaller than :paramref:`threshold`. The revived values are randomly and uniformly
    sampled between the extreme values of :paramref:`targets` across all compartments. The
    range can be scaled by the parameter :paramref:`scaler`. Note that this function does
    not conserve the quantities in :paramref:`targets` across all compartments, since the
    new values are randomly generated.

    Args:
        Js:
            Constant. The 1D array with the size of :math:`N_\mathrm{M}`, containing the relative
            volumes of compartments :math:`J_m`.
        targets:
            Mutable. 2D array with the size of :math:`N_* \times N_\mathrm{M}`, containing the
            values to be revived. The second dimension has to be the same as that of
            :paramref:`Js`. Note that this is not checked.
        threshold:
            Constant. The threshold below which the corresponding compartment is
            considered dead. For each element of :paramref:`Js` smaller than this
            parameter, the corresponding compartment will be considered as dead, and its
            :paramref:`targets` values will then be randomly drawn between the
            corresponding minimum and maximum values of :paramref:`targets` across all
            compartments. Corresponding :paramref:`Js` will be set to be unity.
        rng:
            Mutable. Random number generator for reviving.
        scaler:
            Constant. The scaler for generating random new values.

    Returns:
        : Number of dead compartments that have been revived.
    """
    revive_count = 0
    num_comp, num_part = targets.shape

    target_centers = np.full(num_comp, 0.0, float)
    omega_widths = np.full(num_comp, 0.0, float)
    for itr_component in range(num_comp):
        current_target_max = targets[itr_component].max()
        current_target_min = targets[itr_component].min()
        target_centers[itr_component] = (current_target_max + current_target_min) * 0.5
        omega_widths[itr_component] = (current_target_max - current_target_min) * 0.5

    # revive the compartment with random conjugate field
    for itr_compartment in range(num_part):
        if Js[itr_compartment] <= threshold:
            Js[itr_compartment] = 1.0
            revive_count += 1
            for itr_component in range(num_comp):
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
    r"""Revive dead compartments by copying living ones.

    Revive compartments whose relative volume (element of :paramref:`Js`) is smaller than
    :paramref:`threshold`. The revived values are randomly copied from other living
    compartments. Note that this function conserves the quantities in :paramref:`targets`
    across all compartments by modifying the volumes :paramref:`Js` accordingly.

    Args:
        Js:
            Constant. The 1D array with the size of :math:`N_\mathrm{M}`, containing the relative
            volumes of compartments :math:`J_m`.
        targets:
            Mutable. 2D array with the size of :math:`N_* \times N_\mathrm{M}`, containing the
            values to be revived. The second dimension has to be the same as that of
            :paramref:`Js`. Note that this is not checked.

        threshold:
            Constant. The threshold below which the corresponding compartment is
            considered dead. For each element of :paramref:`Js` smaller than this
            parameter, the corresponding compartment will be considered as dead, and its
            :paramref:`targets` values will then be copied from that of a living
            compartment. At the same time, the corresponding elements (both the dead and
            the copied living one) in :paramref:`Js` will be redistributed to ensure
            conservation of :paramref:`Js`.
        rng:
            Mutable. Random number generator for reviving.

    Returns:
        : Number of revives
    """
    revive_count = 0
    num_comp, num_part = targets.shape

    dead_indexes = np.full(num_part, -1, dtype=np.int32)
    dead_count = 0
    living_nicely_indexes = np.full(num_part, -1, dtype=np.int32)
    living_nicely_count = 0
    for itr_compartment in range(num_part):
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
            living_nicely_count -= 1
            living_nicely_indexes[pos_in_living] = living_nicely_indexes[
                living_nicely_count
            ]
            living_nicely_indexes[living_nicely_count] = -1
    return revive_count


@nb.njit()
def multicomponent_self_consistent_metastep(
    interaction: InteractionBaseCompiled,
    entropy: EntropyBaseCompiled,
    ensemble: EnsembleBaseCompiled,
    constraints: tuple[ConstraintBaseCompiled],
    *,
    omegas: np.ndarray,
    Js: np.ndarray,
    phis_comp: np.ndarray,
    phis_feat: np.ndarray,
    steps_inner: int,
    acceptance_Js: float,
    Js_step_upper_bound: float,
    acceptance_omega: float,
    kill_threshold: float,
    revive_tries: int,
    revive_scaler: float,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, int, bool]:
    r"""
    The core algorithm of finding coexisting states of multicomponent systems with
    self-consistent iterations.

    Args:
        interaction:
            Constant. The compiled interaction instance. See
            :class:`~flory.interaction.base.InteractionBaseCompiled` for more information.
        entropy:
            Constant. The compiled entropy instance. See
            :class:`~flory.entropy.base.EntropyBaseCompiled` for more information.
        ensemble:
            Constant. The compiled ensemble instance. See
            :class:`~flory.ensemble.base.EnsembleBaseCompiled` for more information.
        constraints:
            Constant. The tuple of compiled constraint instance. Note that constraint
            instances are usually stateful, therefore the internal states of
            :paramref:`constraints` are actually mutable. See
            :class:`~flory.constraint.base.constraintBaseCompiled` for more information.
        omegas:
            Mutable. 2D array with size of :math:`N_\mathrm{S} \times N_\mathrm{M}`, containing the
            conjugate field :math:`w_r^{(m)}` of features. Note that this field is both
            used as input and output. Note again that this function DO NOT initialize
            :paramref:`omegas`, it should be initialized externally, and usually a random
            initialization will be a reasonable choice.
        Js:
            Mutable. 1D array with size of :math:`N_\mathrm{M}`, containing the relative volumes of
            the compartments :math:`J_m`. The average value of `Js` will and should be
            unity, in order to keep the values invariant for different :math:`N_\mathrm{M}`. Note
            that this field is both used as input and output. An all-one array is usually
            a nice initialization, unless resume of a previous run is intended.
        phis_comp:
            Output. 2D array with size of :math:`N_\mathrm{C} \times N_\mathrm{M}`, containing the
            volume fractions of components :math:`\phi_i^{(m)}`.
        phis_feat:
            Output. 2D array with size of :math:`N_\mathrm{S} \times N_\mathrm{M}`, containing the
            volume fractions of features :math:`\phi_r^{(m)}`.
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
            :math:`w_r^{(m)}`). This value determines the amount of changes accepted in
            each step for the :math:`w_r^{(m)}` field. Note that if the iteration of
            :math:`J_m` is scaled down due to parameter :paramref:`Js_step_upper_bound`,
            the iteration of :math:`w_r^{(m)}` fields will be scaled down simultaneously.
            Note that this value also scales the evolution of the internal states
            (Lagrange multipliers) of the :paramref:`constraints`. See the documentation
            of actual constraint class for additional acceptances for
            :paramref:`constraints`. Typically this value can take the order of
            :math:`10^{-2}`, or smaller when the system becomes larger or stiffer.
        kill_threshold:
            Constant. The threshold of the :math:`J_m` for a compartment to be considered
            dead and killed afterwards. Should be not less than 0. In each iteration step,
            the :math:`J_m` array will be checked, for each element smaller than this
            parameter, the corresponding compartment will be killed and 0 will be assigned
            to the internal mask. The dead compartment may be revived, depending on whether
            reviving is allowed or whether the number of the revive tries has been
            exhausted.
        revive_tries:
            Constant. Number of tries left to revive the dead compartment. 0 or negative
            value indicates no reviving. When this value is exhausted, i.e. the number of
            revive in current function call exceeds this value, the revive will be turned
            off. Note that this function does not decrease this value, but returns the number
            of revives that have happened after completion.
        revive_scaler:
            Constant. The scaling factor for the conjugate fields :math:`w_r^{(m)}`
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
        [3]: Max absolute constraints error.
        [4]: Number of revives.
        [5]: Whether no phase is killed in the last step.
    """
    num_feat, num_part = omegas.shape

    n_valid_phase = 0

    revive_count = 0
    for _ in range(steps_inner):
        # check if we are still allowed to revive compartments
        if revive_count < revive_tries:
            n_valid_phase = count_valid_compartments(Js, kill_threshold)
            if n_valid_phase != num_part:
                # revive dead compartments
                revive_count += revive_compartments_by_random(
                    Js, omegas, kill_threshold, rng, revive_scaler
                )

        # generate masks for the compartments
        masks = make_valid_compartment_masks(Js, kill_threshold)
        n_valid_phase = int(masks.sum())
        Js *= masks

        # calculate volume fractions, single molecular partition function Q and incompressibility
        Qs = entropy.partition(phis_comp, omegas, Js)  # modifies phis_comp directly
        incomp = ensemble.normalize(phis_comp, Qs, masks)  # modifies phis_comp directly
        entropy.comp_to_feat(phis_feat, phis_comp)  # modifies phis_feat directly
        max_abs_incomp = np.abs(incomp).max()

        # prepare constraints: constraints are stateful
        if constraints:
            for cons in literal_unroll(constraints):
                cons.prepare(phis_feat, Js, masks)

        # temp for omega, namely chi.phi
        omega_temp = interaction.potential(phis_feat)

        # xi, the Lagrange multiplier
        xi = interaction.incomp_coef(phis_feat) * incomp
        for itr_feat in range(num_feat):
            xi += omegas[itr_feat] - omega_temp[itr_feat]
        for cons in literal_unroll(constraints):
            for itr_feat in range(num_feat):
                xi -= cons.potential[
                    itr_feat
                ]  # potential from constraints are already calculated in preparation.
        xi *= masks
        xi /= num_feat

        # local energy. i.e. energy of phases excluding the partition function part
        local_energy = (
            interaction.volume_derivative(omega_temp, phis_feat)
            + entropy.volume_derivative(phis_comp)
            + xi * incomp
        )
        for cons in literal_unroll(constraints):
            local_energy += (
                cons.volume_derivative
            )  # volume_derivative from constraints are already calculated in preparation.
            omega_temp += cons.potential
        for itr_feat in range(num_feat):
            omega_temp[itr_feat] += xi
            local_energy -= omega_temp[itr_feat] * phis_feat[itr_feat]

        # calculate the difference of Js
        local_energy_mean = (local_energy * Js).sum() / n_valid_phase
        Js_diff = (local_energy_mean - local_energy) * masks
        max_abs_Js_diff = np.abs(Js_diff).max()

        # calculate additional factor to scale down iteration
        Js_max_change = max(max_abs_Js_diff * acceptance_Js, Js_step_upper_bound)
        additional_factor = Js_step_upper_bound / Js_max_change

        # update Js
        Js += additional_factor * acceptance_Js * Js_diff
        Js *= masks
        Js += 1 - (Js.sum() / n_valid_phase)
        Js *= masks

        # calculate difference of omega and update omega directly
        max_abs_omega_diff = 0
        for itr_comp in range(num_feat):
            omega_temp[itr_comp] -= omegas[itr_comp]
            omega_temp[itr_comp] *= masks
            max_abs_omega_diff = max(max_abs_omega_diff, omega_temp[itr_comp].max())
            omegas[itr_comp] += (
                additional_factor * acceptance_omega * omega_temp[itr_comp]
            )
            omegas[itr_comp] *= masks

        max_constraint_residue = 0
        for cons in literal_unroll(constraints):
            max_constraint_residue = max(
                max_constraint_residue,
                cons.evolve(additional_factor * acceptance_omega, masks),
            )

    # count the valid phases in the last step
    n_valid_phase_last = count_valid_compartments(Js, kill_threshold)

    return (
        max_abs_incomp,
        max_abs_omega_diff,
        max_abs_Js_diff,
        max_constraint_residue,
        revive_count,
        n_valid_phase == n_valid_phase_last,
    )
