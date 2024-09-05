"""Module for a general finder for coexisting phases.

:mod:`flory.mcmp` provides the general tool for finding equilibrium multiple coexisting
phases in multicomponent mixtures with defined :mod:`~flory.interaction`,
:mod:`~flory.entropy`, :mod:`~flory.ensemble` and :mod:`~flory.constraint`. The finder is
provided through the class :class:`CoexistingPhasesFinder`, which is designed to be
flexible, reusable, independent and efficient: 

- flexible: :class:`CoexistingPhasesFinder` can be applied to different interaction,
  entropy, ensemble and constraint, as soon as the they implement the a minimal set of
  methods. 

- reusable: :class:`CoexistingPhasesFinder` tries every possibility to avoid recreation of
  objects as soon as the system size is unchanged, making it ideal to be reused for
  different parameters. 

- independent: :class:`CoexistingPhasesFinder` owns all the data it needs once created.
  Therefore multiple instances can be created and stored freely. 

- efficient: :class:`CoexistingPhasesFinder` use :func:`numba.jit` to compile all of its
  core algorithms.

The usage of :class:`CoexistingPhasesFinder` usually follows the creation-and-run manner
for new instance, or reinitialization-and-run for existing instance. The reinitialization
might be skipped in the case that the previous result in the existing instance provides
good initial guess for next run, which is a typical case when constructing phase diagram. 

See :ref:`Examples` for examples.

.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from datetime import datetime
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from ..common import *
from ..constraint import ConstraintBase, ConstraintBaseCompiled, NoConstraintCompiled
from ..ensemble import EnsembleBase, EnsembleBaseCompiled
from ..entropy import EntropyBase, EntropyBaseCompiled
from ..interaction import InteractionBase, FloryHugginsInteractionCompiled
from ._finder_impl import *


class CoexistingPhasesFinder:
    r"""Class for a general finder of coexisting phases.

    This class is recommended when multiple instances of :paramref:`interaction`,
    :paramref:`entropy`, :paramref:`ensemble` or :paramref:`constraints` need to be
    calculated. The class will reuse all the options and the internal resources. Note that
    reuse the instance of this class is only possible when all the system sizes are not
    changed, including the number of components :math:`N_\mathrm{C}`, the number of
    features :math:`N_\mathrm{S}` and the number of compartments :math:`N_\mathrm{M}`.
    """

    def __init__(
        self,
        interaction: InteractionBase,
        entropy: EntropyBase,
        ensemble: EnsembleBase,
        constraints: ConstraintBase | tuple[ConstraintBase] | None = None,
        *,
        num_part: int | None = None,
        rng: np.random.Generator | None = None,
        max_steps: int = 1000000,
        convergence_criterion: str = "standard",
        tolerance: float = 1e-5,
        interval: int = 10000,
        progress: bool = True,
        random_std: float = 5.0,
        acceptance_Js: float = 0.0002,
        acceptance_omega: float = 0.002,
        Js_step_upper_bound: float = 0.001,
        kill_threshold: float = 0.0,
        revive_scaler: float = 1.0,
        max_revive_per_compartment: int = 16,
        **kwargs,
    ):
        r"""
        Args:
            interaction:
                The interaction instance that can provide a compiled interaction instance.
                See :class:`~flory.interaction.base.InteractionBase` for more information.
            entropy:
                The entropy instance that can provide a compiled entropy instance. See
                :class:`~flory.entropy.base.EntropyBase` for more information.
            ensemble:
                The ensemble instance that can provide a compiled ensemble instance. See
                :class:`~flory.ensemble.base.EnsembleBase` for more information.
            constraints:
                The constraint instance or a list of constrain instances, each of which
                can provide a compiled constraint instance. See
                :class:`~flory.constraint.base.ConstraintBase` for more information.
            num_part:
                Number of compartments :math:`N_\mathrm{M}` in the system. By default this is set to
                be :math:`8 N_\mathrm{C}`.
            rng:
                Random number generator for initialization and reviving. None indicates
                that a new random number generator should be created by the class, seeded
                by current timestamp.
            max_steps:
                The default maximum number of steps in each run to find the coexisting
                phases. This value can be temporarily overwritten, see :meth:`run` for
                more information.
            convergence_criterion:
                The criterion to determine convergence. Currently "standard" is the only
                option, which requires checking of incompressibility, field error and the
                volume error. Note that all these metrics are state functions, namely they
                are independent of iteration parameters.
            tolerance:
                The tolerance to determine convergence. This value can be temporarily
                overwritten. See :paramref:`convergence_criterion` and :meth:`run` for
                more information.
            interval:
                The interval of steps to check convergence. This value can be temporarily
                overwritten, see :meth:`run` for more information.
            progress:
                Whether to show progress bar when checking convergence. This value can be
                temporarily overwritten, see :meth:`run` for more information.
            random_std:
                The amplitude of the randomly generated conjugate fields internally.
                During instantiation, the conjugate field is randomly generated according
                to an normal distribution with standard deviation :paramref:`random_std`.
            acceptance_Js:
                The acceptance of the relative compartment size :math:`J_m`. This value
                determines the amount of changes accepted in each step for the :math:`J_m`
                field. Typically this value can take the order of :math:`10^{-3}`, or
                smaller when the system becomes larger or stiffer.
            Js_step_upper_bound:
                The maximum change of the relative compartment size :math:`J_m` per step.
                This value is designed to reduce the risk that a the volume of a
                compartment changes too fast before it develops meaningful composition. If
                the intended change is larger this value, all the changes will be scaled
                down to guarantee that the maximum changes do not exceed this value.
                Typically this value can take the order of :math:`10^{-3}`, or smaller
                when the system becomes larger or stiffer.
            acceptance_omega:
                The acceptance of the conjugate fields :math:`w_r^{(m)}`. This value
                determines the amount of changes accepted in each step for the
                :math:`w_r^{(m)}` field. Note that if the iteration of :math:`J_m` is
                scaled down due to parameter :paramref:`Js_step_upper_bound`, the
                iteration of :math:`w_r^{(m)}` fields will be scaled down simultaneously.
                Typically this value can take the order of :math:`10^{-2}`, or smaller
                when the system becomes larger or stiffer.
            kill_threshold:
                The threshold of the :math:`J_m` for a compartment to be considered dead
                and killed afterwards. Should be not less than 0. In each iteration step,
                the :math:`J_m` array will be checked, for each element smaller than this
                parameter, the corresponding compartment will be killed and 0 will be
                assigned to the internal mask. The dead compartment may be revived,
                depending whether reviving is allowed or whether the number of the revive
                tries has been exhausted.
            revive_scaler:
                The scaler for the value of the newly-generated conjugate fields when a
                dead compartment is revived. The compartment is revived by drawing random
                numbers for their conjugate fields in the range of the minimum and the
                maximum of the their conjugate fields :math:`w_r^{(m)}` across all
                compartments. This value determines whether this range should be enlarged
                (a value larger than 1) or reduced (a value smaller than 1). Typically 1.0
                or a value slightly larger than 1.0 will be a reasonable choice.
            max_revive_per_compartment:
                Maximum average number of tries per compartment to revive the dead
                compartments. 0 or negative value indicates no reviving. When this value
                is exhausted, the revive will be turned off.
            kwargs:
                Additional keyword arguments that will be forwarded to the compile method
                in :paramref:`interaction`, :paramref:`entropy`, :paramref:`ensemble` or
                :paramref:`constraints`. Redundant arguments are allowed and ignored.
        """
        self._logger = logging.getLogger(self.__class__.__name__)

        self._interaction = interaction.compiled(**kwargs)
        self._entropy = entropy.compiled(**kwargs)
        self._ensemble = ensemble.compiled(**kwargs)

        self._num_comp = self._entropy.num_comp
        self._num_feat = self._entropy.num_feat
        self._num_part = num_part if num_part else self._num_comp * 8

        self._constraints = []
        if constraints:
            if isinstance(constraints, ConstraintBase):
                self._constraints.append(constraints.compiled(**kwargs))
            elif isinstance(constraints, Iterable):
                for cons in constraints:
                    self._constraints.append(cons.compiled(**kwargs))
        else:
            self._constraints.append(NoConstraintCompiled(self._num_feat))

        self.check_instance(self._interaction)
        self.check_instance(self._entropy)
        self.check_instance(self._ensemble)
        for cons in self._constraints:
            self.check_instance(cons)

        # rng
        if rng is None:
            self._rng_is_external = False
            self._rng_seed = int(datetime.now().timestamp())
            self._rng = np.random.default_rng(self._rng_seed)
        else:
            self._rng_is_external = True
            self._rng_seed = 0
            self._rng = rng

        # other parameters
        self._max_steps = max_steps
        self._convergence_criterion = convergence_criterion
        self._tolerance = tolerance
        self._interval = interval
        self._progress = progress

        self._random_std = random_std
        self._acceptance_Js = acceptance_Js
        self._acceptance_omega = acceptance_omega
        self._Js_step_upper_bound = Js_step_upper_bound
        self._kill_threshold = kill_threshold
        self._revive_scaler = revive_scaler
        self._max_revive_per_compartment = max_revive_per_compartment

        # diagnostics
        self._diagnostics: dict[str, Any] = {}

        ## initialize derived internal states
        self._Js = np.full(self._num_part, 0.0, float)
        self._omegas = np.full((self._num_feat, self._num_part), 0.0, float)
        self._phis_feat = np.full((self._num_feat, self._num_part), 0.0, float)
        self._phis_comp = np.full((self._num_comp, self._num_part), 0.0, float)
        self._revive_count_left = self._max_revive_per_compartment * self._num_part
        self._kwargs_for_instances = kwargs
        self.reinitialize_random()

    def check_instance(
        self,
        compiled_instance: (
            InteractionBaseCompiled
            | EntropyBaseCompiled
            | EnsembleBaseCompiled
            | ConstraintBaseCompiled
        ),
    ) -> None:
        """Check the size of the compiled instance.

        This method checks whether the compiled instance from
        :class:`~flory.interaction.base.InteractionBase`,
        :class:`~flory.entropy.base.EntropyBase`,
        :class:`~flory.ensemble.base.EnsembleBase` or
        :class:`~flory.constraint.base.ConstraintBase` has the correct system size with
        the finder. An exception will be raised on failure.

        Args:
            compiled_instance:
                The instance to check.
        """
        if (
            hasattr(compiled_instance, "num_comp")
            and compiled_instance.num_comp != self._num_comp
        ):
            self._logger.error(
                "number of components %d obtained from compiled objects is "
                "incompatible, %d is already set for the finder.",
                compiled_instance.num_comp,
                self._num_comp,
            )
            raise ComponentNumberError
        if (
            hasattr(compiled_instance, "num_feat")
            and compiled_instance.num_feat != self._num_feat
        ):
            self._logger.error(
                "number of features %d obtained from compiled objects is "
                "incompatible, %d is already set for the finder.",
                compiled_instance.num_feat,
                self._num_feat,
            )
            raise FeatureNumberError

    def check_field(self, field: np.ndarray) -> np.ndarray:
        r"""Check the size of a field.

        This method checks whether the :paramref:`field` has the same size as the
        :attr:`omegas`, which contains the conjugate field of the volume fractions of the
        features, :math:`w_r^{(m)}`, which has the size of :math:`N_\mathrm{S} \times N_\mathrm{M}`.
        An exception will be raise on failure.

        Args:
            field :
                The field to check.

        Returns:
            : The field converted to numpy array.
        """
        field = np.array(field)
        if field.shape != self._omegas.shape:
            self._logger.error(
                "field with size of %s is invalid. It must have the size of %s.",
                field.shape,
                (self._num_feat, self._num_part),
            )
            raise ValueError("New field must match the size of the old one.")
        return field

    def set_interaction(
        self, interaction: InteractionBase, *, if_reset_revive: bool = True, **kwargs
    ) -> None:
        """Set a new interaction instance.

        This method sets a new interaction instance, using the updated keyword arguments.
        Note that this method does not change the default.

        Args:
            interaction:
                New interaction instance.
            if_reset_revive:
                Whether the revive count is reset.
            kwargs:
                The keyword arguments to update the default.
        """
        self._interaction = interaction.compiled(
            **{**self._kwargs_for_instances, **kwargs}
        )
        self.check_instance(self._interaction)
        if if_reset_revive:
            self.reset_revive()

    def set_entropy(
        self, entropy: EntropyBase, *, if_reset_revive: bool = True, **kwargs
    ) -> None:
        """Set a new entropy instance.

        This method sets a new entropy instance, using the updated keyword arguments. Note
        that this method does not change the default.

        Args:
            entropy:
                New entropy instance.
            if_reset_revive:
                Whether the revive count is reset.
            kwargs:
                The keyword arguments to update the default.
        """

        self._entropy = entropy.compiled(**{**self._kwargs_for_instances, **kwargs})
        self.check_instance(self._entropy)
        if if_reset_revive:
            self.reset_revive()

    def set_ensemble(
        self, ensemble: EnsembleBase, *, if_reset_revive: bool = True, **kwargs
    ) -> None:
        """Set a new ensemble instance.

        This method sets a new ensemble instance, using the updated keyword arguments.
        Note that this method does not change the default.

        Args:
            ensemble:
                New ensemble instance.
            if_reset_revive:
                Whether the revive count is reset.
            kwargs:
                The keyword arguments to update the default.
        """

        self._ensemble = ensemble.compiled(**{**self._kwargs_for_instances, **kwargs})
        self.check_instance(self._ensemble)
        if if_reset_revive:
            self.reset_revive()

    def set_constraints(
        self,
        constraints: ConstraintBase | tuple[ConstraintBase] | None = None,
        *,
        if_reset_revive: bool = True,
        kwargs_individual: dict | tuple[dict] | None = None,
        **kwargs,
    ) -> None:
        """Set a new set of constraint instances.

        This method sets a new set of constraint instances, using the updated keyword
        arguments. Note that this method does not change the default.

        Args:
            constraints:
                New set of constraint instances.
            if_reset_revive:
                Whether the revive count is reset.
            kwargs_individual:
                The keyword arguments to update the default for each constraint. This
                parameter has higher priority than :paramref:`kwargs`.
            kwargs:
                The keyword arguments to update the default.
        """

        self._constraints = []
        if constraints:
            if isinstance(constraints, ConstraintBase):
                if kwargs_individual is None:
                    kwargs_individual = {}
                self._constraints.append(
                    constraints.compiled(
                        **{**self._kwargs_for_instances, **kwargs, **kwargs_individual}
                    )
                )
            elif isinstance(constraints, Iterable):
                if kwargs_individual is None:
                    kwargs_individual = [{}] * len(constraints)
                for cons, current_args in zip(constraints, kwargs_individual):
                    self._constraints.append(
                        cons.compiled(
                            **{**self._kwargs_for_instances, **kwargs, **current_args}
                        )
                    )
        else:
            self._constraints.append(NoConstraintCompiled(self._num_feat))

        for cons in self._constraints:
            self.check_instance(cons)

        self.reinitialize_constraint()

        if if_reset_revive:
            self.reset_revive()

    def reset_revive(self):
        """Reset the internal revive count."""

        self._revive_count_left = self._max_revive_per_compartment * self._num_part

    def reinitialize_constraint(self):
        """Reinitialize the constraints"""

        for cons in self._constraints:
            cons.initialize(self._num_part)

    def reinitialize_random(self):
        """Reinitialize :math:`w_r^{(m)}` randomly.

        See parameter :paramref:`CoexistingPhasesFinder.random_std` for more information.
        """
        self._omegas = self._rng.normal(
            0.0,
            self._random_std,
            (self._num_feat, self._num_part),
        )
        self._Js = np.full(self._num_part, 1.0, float)
        self.reset_revive()
        self.reinitialize_constraint()

    def reinitialize_from_omegas(self, omegas: np.ndarray):
        r"""Reinitialize :math:`w_r^{(m)}` from input.

        Args:
            omegas:
                New :math:`w_r^{(m)}` field, must have the size of :math:`N_\mathrm{S}
                \times M`.
        """
        self._omegas = self.check_field(omegas)
        self._Js = np.ones_like(self._Js)
        self.reset_revive()
        self.reinitialize_constraint()

    def reinitialize_from_phis(self, phis: np.ndarray):
        r"""Reinitialize :math:`w_r^{(m)}` from :math:`\phi_r^{(m)}`.

        The conjugated fields :math:`w_r^{(m)}` are initialized from volume fraction
        fields :math:`\phi_r^{(m)}`. Note that it is not guaranteed that the initial
        volume fractions :math:`\phi_r^{(m)}` are fully respected. The input is only
        considered as a suggestion for the generation of :math:`w_r^{(m)}` field.

        Args:
            phis:
                New :math:`\phi_r^{(m)}`, must have the size of :math:`N_\mathrm{S} \times
                M`.
        """
        phis = self.check_field(phis)
        self._omegas = self._interaction.potential(phis)
        self._Js = np.ones_like(self._Js)
        self.reset_revive()
        self.reinitialize_constraint()

    @property
    def omegas(self) -> np.ndarray:
        r"""Internal conjugate fields :math:`w_r^{(m)}`.

        Read-only array of length :math:`N_\mathrm{S} \times N_\mathrm{M}`. Use
        :meth:`reinitialize_from_omegas` to initialize the system from given
        :math:`w_r^{(m)}`.
        """
        return self._omegas

    @property
    def diagnostics(self) -> dict:
        """Diagnostic information  available after :meth:`run` finished.

        The diagnostics dictionary contains the convergence status and the original volume
        fractions before the clustering and sorting algorithm is used to determine the
        unique phases.
        """
        return self._diagnostics

    def run(
        self,
        *,
        max_steps: float | None = None,
        tolerance: float | None = None,
        interval: int | None = None,
        progress: bool | None = None,
    ) -> Phases:
        r"""Run instance to find coexisting phases.

        The keywords arguments can be used to temporarily overwrite the provided values
        during construction of the class. Note that this temporary values will not affect
        the defaults. See class :class:`CoexistingPhasesFinder` for other tunable
        parameters. After each call, the diagnostics will be updated, which can be used
        to, for example, inspecting whether the iteration really converges. See
        :attr:`diagnostics` for more information.

        Args:
            max_steps:
                The maximum number of steps to find the coexisting phases.
            tolerance:
                The tolerance to determine convergence. See
                :paramref:`~CoexistingPhasesFinder.convergence_criterion` for more
                information.
            interval:
                The interval of steps to check convergence.
            progress:
                Whether to show progress bar when checking convergence.

        Returns:
            :   Composition and relative size of the compartments. The member
                :paramref:`~flory.common.phases.Phases.volumes` (accessible by
                :code:`.volumes`) contains the fraction of volume of each compartment. The
                member :paramref:`~flory.common.phases.Phases.fractions` (accessible by
                :code:`.fractions`) contains volume fractions of all components. Use method
                :meth:`~.flory.common.phases.Phases.get_clusters` to obtain unique phases.
        """
        if max_steps is None:
            max_steps = self._max_steps
        if tolerance is None:
            tolerance = self._tolerance
        if interval is None:
            interval = self._interval
        if progress is None:
            progress = self._progress

        steps_tracker = int(np.ceil(max_steps / interval))
        steps_inner = max(1, int(np.ceil(max_steps)) // steps_tracker)

        steps = 0

        bar_max = -np.log10(tolerance)
        bar_format = "{desc:<20}: {percentage:3.0f}%|{bar}{r_bar}"
        bar_args = {
            "total": bar_max,
            "disable": not progress,
            "bar_format": bar_format,
        }
        bar_text_args = {
            "total": 1,
            "disable": not progress,
            "bar_format": "{desc}",
        }
        pbar1 = tqdm(**bar_args, position=0, desc="Incompressibility")
        pbar2 = tqdm(**bar_args, position=1, desc="Field Error")
        pbar3 = tqdm(**bar_args, position=2, desc="Volume Error")
        pbar4 = tqdm(**bar_args, position=3, desc="Constraint Residue")
        pbar5 = tqdm(**bar_text_args, position=4, desc="Revive Count Left")

        bar_val_func = lambda a: max(0, min(round(-np.log10(max(a, 1e-100)), 1), bar_max))

        start_time = time.time()

        for _ in range(steps_tracker):
            # do the inner steps
            (
                max_abs_incomp,
                max_abs_omega_diff,
                max_abs_Js_diff,
                max_constraint_residue,
                revive_count,
                is_last_step_safe,
            ) = multicomponent_self_consistent_metastep(
                self._interaction,
                self._entropy,
                self._ensemble,
                tuple(self._constraints),
                omegas=self._omegas,
                Js=self._Js,
                phis_feat=self._phis_feat,
                phis_comp=self._phis_comp,
                steps_inner=steps_inner,
                acceptance_Js=self._acceptance_Js,
                Js_step_upper_bound=self._Js_step_upper_bound,
                acceptance_omega=self._acceptance_omega,
                kill_threshold=self._kill_threshold,
                revive_tries=self._revive_count_left,
                revive_scaler=self._revive_scaler,
                rng=self._rng,
            )

            steps += steps_inner
            self._revive_count_left -= revive_count

            if progress:
                pbar1.n = bar_val_func(max_abs_incomp)
                pbar2.n = bar_val_func(max_abs_omega_diff)
                pbar3.n = bar_val_func(max_abs_Js_diff)
                pbar4.n = bar_val_func(max_constraint_residue)
                pbar5.n = 1
                pbar1.refresh()
                pbar2.refresh()
                pbar3.refresh()
                pbar4.refresh()
                pbar5.set_description_str(
                    "{:<20}: {}".format("Revive Count Left", self._revive_count_left)
                )
                pbar5.refresh()

            # check convergence
            if self._convergence_criterion == "standard":
                if (
                    is_last_step_safe
                    and tolerance > max_abs_incomp
                    and tolerance > max_abs_omega_diff
                    and tolerance > max_abs_Js_diff
                    and tolerance > max_constraint_residue
                ):
                    self._logger.info(
                        "Composition and volumes reached stationary state after %d steps",
                        steps,
                    )
                    break
            else:
                raise ValueError(
                    f"Undefined convergence criterion: {self._convergence_criterion}"
                )

        pbar1.close()
        pbar2.close()
        pbar3.close()
        pbar4.close()
        pbar5.close()

        # get final result
        final_Js = self._Js.copy()
        final_phis_comp = self._phis_comp.copy()
        revive_compartments_by_copy(
            Js=final_Js,
            targets=final_phis_comp,
            threshold=self._kill_threshold,
            rng=self._rng,
        )

        n_valid = count_valid_compartments(self._Js, self._kill_threshold)
        if n_valid < self._num_part // 2:
            self._logger.warning(
                "Only %d out of %d compartments are living, the result might not be reliable.",
                n_valid,
                self._num_part,
            )

        # store diagnostic output
        self._diagnostics = {
            "steps": steps,
            "time": time.time() - start_time,
            "max_abs_incomp": max_abs_incomp,
            "max_abs_omega_diff": max_abs_omega_diff,
            "max_abs_js_diff": max_abs_Js_diff,
            "max_constraint_residue": max_constraint_residue,
            "revive_count_left": self._revive_count_left,
            "phis": self._phis_comp.copy(),
            "Js": self._Js.copy(),
        }

        # transpose phi since `Phases` uses a different convention
        return Phases(final_Js, final_phis_comp.T)
