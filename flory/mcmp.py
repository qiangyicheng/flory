"""Module for finding coexisting phases of multicomponent mixtures.

:mod:`flory.mcmp` provides tools for finding equilibrium multiple coexisting phases in
multicomponent mixtures in the canonical ensemble based on Flory-Huggins theory. The
module provides both the function API :meth:`find_coexisting_phases` and the class API
:class:`CoexistingPhasesFinder`. The function :meth:`find_coexisting_phases` aims at
easing the calculation of the coexisting phases with a single system setting, namely one
single point in the phase diagram. In contrast, The class :class:`CoexistingPhasesFinder`
is designed for reuse over systems with the same system sizes such as the number of
components, which includes generating a coexisting curve or sampling a phase diagram. It
creates a finder to maintain its internal data, and provides more control and diagnostics
over the iteration. See :ref:`Examples` for examples.

.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import logging
import numpy as np
from datetime import datetime
from typing import Optional
from tqdm.auto import tqdm
from .detail.mcmp_impl import *


class CoexistingPhasesFinder:
    """
    Create a finder for coexisting phases.
    """

    def __init__(
        self,
        chis: np.ndarray,
        phi_means: np.ndarray,
        num_compartments: int,
        *,
        sizes: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
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
        additional_chis_shift: float = 1.0,
    ):
        """
        Construct a :class:`CoexistingPhasesFinder` instance for finding coexisting
        phases. This class is recommended when multiple instances of :paramref:`chis`
        matrix or :paramref:`phi_means` vector need to be calculated. The class will reuse
        all the options and the internal resources. Note that reuse the instance of this
        class is only possible when all the system sizes are not changed, including the
        number of components :math:`N_\\mathrm{c}` and the number of compartments
        :math:`M`. The the number of components :math:`N_\\mathrm{c}` is inferred from
        :paramref:`chis` and :paramref:`phi_means`, while :math:`N_\\mathrm{c}` is set by
        the parameter :paramref:`num_compartments`. Setting :paramref:`chis` matrix and
        :paramref:`phi_means` manually by the setters leads to the reset of the internal
        revive counters.

        Args:
            chis:
                The interaction matrix. 2D array with size of :math:`N_\\mathrm{c} \\times
                N_\\mathrm{c}`. This matrix should be the full :math:`\\chi_{ij}` matrix
                of the system, including the solvent component. Note that the matrix must
                be symmetric, which is not checked but should be guaranteed externally.
            phi_means:
                The average volume fractions :math:`\\bar{\\phi}_i` of all the components
                of the system. 1D array with size of :math:`N_\\mathrm{c}`. Note that the
                volume fraction of the solvent is included as well, therefore the sum of
                this array must be unity, which is not checked by this function and should
                be guaranteed externally.
            num_compartments:
                Number of compartments :math:`M` in the system.
            sizes:
                The relative molecule volumes :math:`l_i` of the components. 1D array with
                size of :math:`N_\\mathrm{c}`. This sizes vector should be the full sizes
                vector of the system, including the solvent component. An element of one
                indicates that the corresponding specie has the same volume as the
                reference. None indicates a all-one vector.
            rng:
                Random number generator for initialization and reviving. None indicates
                that a new random number generator should be created by the class, seeded
                by current timestamp.
            max_steps:
                The default maximum number of steps in each run to find the coexisting
                phases. This value can be temporarily overwritten, see :meth:`run` for more
                information.
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
                The acceptance of the conjugate fields :math:`\\omega_i^{(m)}`. This value
                determines the amount of changes accepted in each step for the
                :math:`\\omega_i^{(m)}` field. Note that if the iteration of :math:`J_m` is
                scaled down due to parameter :paramref:`Js_step_upper_bound`, the
                iteration of :math:`\\omega_i^{(m)}` fields will be scaled down simultaneously.
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
                maximum of the :math:`\\omega_i^{(m)}` their conjugate fields across all
                compartments. This value determines whether this range should be enlarged
                (a value larger than 1) or reduced (a value smaller than 1). Typically 1.0
                or a value slightly larger than 1.0 will be a reasonable choice.
            max_revive_per_compartment:
                Maximum average number of tries per compartment to revive the dead
                compartments. 0 or negative value indicates no reviving. When this value
                is exhausted, the revive will be turned off.
            additional_chis_shift:
                Shift of the entire chis matrix to improve the convergence by evolving
                towards incompressible system faster. This value should be larger than 0.
                Note that with very large value, the convergence will be slowed down,
                since the algorithm no longer have enough ability to temporarily relax the
                incompressibility.
        """
        self._logger = logging.getLogger(self.__class__.__name__)

        # chis
        chis = np.array(chis)
        if chis.shape[0] == chis.shape[1]:
            self._chis = chis
            self._num_components = chis.shape[0]
            self._logger.info(
                f"We infer that there are {self._num_components} components in the system from the chis matrix."
            )
        else:
            self._logger.error(f"chis matrix with size of {chis.shape} is not square.")
            raise ValueError("chis matrix must be square.")
        if (chis != chis.transpose()).any():
            self._logger.error(f"chis matrix is not symmetric.")
            raise ValueError("chis matrix must be symmetric.")

        # phi_means
        phi_means = np.array(phi_means)
        if phi_means.shape[0] == self._num_components:
            self._phi_means = phi_means
            if np.abs(self._phi_means.sum() - 1.0) > 1e-12:
                self._logger.warning(
                    f"Total phi_means is not 1.0. Iteration may never converge."
                )
        else:
            self._logger.error(
                f"phi_means vector with size of {phi_means.shape} is invalid, since {self._num_components} is defined by chis matrix."
            )
            raise ValueError(
                "phi_means vector must imply same component number as chis matrix."
            )

        self._num_compartments = num_compartments

        ## optional arguments

        # sizes
        if sizes is None:
            self._sizes = np.ones(self._num_components)
        else:
            sizes = np.array(sizes)
            if sizes.shape[0] == self._num_components:
                self._sizes = sizes
                if np.sum(self._sizes <= 0):
                    self._logger.warning(
                        f"Non-positive sizes detected. Iteration will probably fail."
                    )
            else:
                self._logger.error(
                    f"sizes vector with size of {sizes.shape} is invalid, since {self._num_components} is defined by chis matrix."
                )
                raise ValueError(
                    "sizes vector must imply same component number as chis matrix."
                )

        # rng
        if rng is None:
            self._rng_is_external = False
            self._rng_seed = int(datetime.now().timestamp())
            self._rng = np.random.default_rng(self._rng_seed)
        else:
            self._rng_is_external = True
            self._rng_seed = int(0)
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
        self._additional_chis_shift = additional_chis_shift

        # diagnostics
        self._diagnostics = {}

        ## initialize derived internal states
        self._Js = np.full(self._num_compartments, 0.0, float)
        self._omegas = np.full((self._num_components, self._num_compartments), 0.0, float)
        self._phis = np.full((self._num_components, self._num_compartments), 0.0, float)
        self._revive_count_left = (
            self._max_revive_per_compartment * self._num_compartments
        )
        self.reinitialize_random()

    def reinitialize_random(self):
        """
        Reinitialize the internal conjugate field :math:`\\omega_i^{(m)}` randomly. See parameter
        :paramref:`CoexistingPhasesFinder.random_std` for more information.
        """
        self._omegas = self._rng.normal(
            0.0,
            self._random_std,
            (self._num_components, self._num_compartments),
        )
        self._Js = np.full(self._num_compartments, 1.0, float)
        self._revive_count_left = (
            self._max_revive_per_compartment * self._num_compartments
        )

    def reinitialize_from_omegas(self, omegas: np.ndarray):
        """
        Reinitialize the internal conjugate field :math:`\\omega_i^{(m)}` from input.

        Args:
            omegas:
                New :math:`\\omega_i^{(m)}` field, must have the same size of
                :math:`N_\\mathrm{c} \\times M`.
        """
        omegas = np.array(omegas)
        if omegas.shape == self._omegas.shape:
            self._omegas = omegas
        else:
            self._logger.error(
                f"new omegas with size of {omegas.shape} is invalid. It must have the size of {(self._num_components, self._num_compartments)}."
            )
            raise ValueError("New omegas must match the size of the old one.")
        self._Js = np.ones_like(self._Js)
        self._revive_count_left = (
            self._max_revive_per_compartment * self._num_compartments
        )

    def reinitialize_from_phis(self, phis):
        """
        Reinitialize the internal conjugate field :math:`\\omega_i^{(m)}` from volume fraction
        field :math:`\\phi_i^{(m)}`. Note that it is not guaranteed that the initial volume
        fraction field :math:`\\phi_i^{(m)}` is fully respected. The input is only considered
        as a suggestion for the generation of :math:`\\omega_i^{(m)}` field.

        Args:
            phis:
                New :math:`\\phi_i^{(m)}` field, must have the same size of :math:`N_\\mathrm{c}
                \\times M`.
        """
        if phis.shape == self._omegas.shape:
            self._omegas = -np.log(phis)
            for itr in range(self._num_components):
                self._omegas[itr] /= self._sizes[itr]
        else:
            self._logger.error(
                f"phis with size of {phis.shape} is invalid. It must have the size of {(self._num_components, self._num_compartments)}."
            )
            raise ValueError("phis must match the size of the omegas.")
        self._Js = np.ones_like(self._Js)
        self._revive_count_left = (
            self._max_revive_per_compartment * self._num_compartments
        )

    @property
    def chis(self) -> np.ndarray:
        """
        The full interaction matrix :math:`\\chi_{ij}`, with the size of
        :math:`N_\\mathrm{c} \\times N_\\mathrm{c}`. Resetting this property requires
        that the new matrix has the same size with the internal one. Note that this
        implies implicit reset of the internal data, the number of revives, but not the
        others including volume fractions :math:`\\phi_i^{(m)}` and conjugate fields
        :math:`\\omega_i^{(m)}`. See class parameters
        :paramref:`~CoexistingPhasesFinder.chis` and
        :paramref:`~CoexistingPhasesFinder.max_revive_per_compartment` for more
        information.
        """
        return self._chis

    @chis.setter
    def chis(self, chis_new: np.ndarray):
        chis_new = np.array(chis_new)
        if chis_new.shape == self._chis.shape:
            if (chis_new != chis_new.transpose()).any():
                self._logger.error(f"chis matrix is not symmetric.")
                raise ValueError("chis matrix must be symmetric.")
            self._chis = chis_new
        else:
            self._logger.error(
                f"new chis with size of {chis_new.shape} is invalid. It must have the size of {self._chis.shape}."
            )
            raise ValueError("New chis must match the size of the old one.")
        self._revive_count_left = (
            self._max_revive_per_compartment * self._num_compartments
        )

    @property
    def phi_means(self) -> np.ndarray:
        """
        The average volume fractions of all components :math:`\\bar{\\phi}_i`, with the
        size of :math:`N_\\mathrm{c}`. Resetting this property requires that the new array
        has the same size with the internal one. Note that this implies implicit reset of
        the internal data, the number of revives, but not the others including volume
        fractions :math:`\\phi_i^{(m)}` and conjugate fields :math:`\\omega_i^{(m)}`. See
        class parameters :paramref:`~CoexistingPhasesFinder.phis_mean` and
        :paramref:`~CoexistingPhasesFinder.max_revive_per_compartment` for more
        information.
        """
        return self._phi_means

    @phi_means.setter
    def phi_means(self, phi_means_new):
        phi_means_new = np.array(phi_means_new)
        if phi_means_new.shape == self._phi_means.shape:
            self._phi_means = phi_means_new
            if np.abs(self._phi_means.sum() - 1.0) > 1e-12:
                self._logger.warning(
                    f"Total phi_means is not 1.0. Iteration may never converge."
                )
        else:
            self._logger.error(
                f"new phi_means with size of {phi_means_new.shape} is invalid. It must have the size of {self._phi_means.shape}."
            )
            raise ValueError("New phi_means must match the size of the old one.")
        self._revive_count_left = (
            self._max_revive_per_compartment * self._num_compartments
        )

    @property
    def sizes(self) -> np.ndarray:
        """
        The relative molecule sizes of all components :math:`l_i` with the size of
        :math:`N_\\mathrm{c}`. Resetting this property requires that the new array has
        the same size with the internal one. Note that this implies implicit reset of the
        internal data, the number of revives, but not the others including volume
        fractions :math:`\\phi_i^{(m)}` and conjugate fields :math:`\\omega_i^{(m)}`. See
        class parameters :paramref:`~CoexistingPhasesFinder.sizes` and
        :paramref:`~CoexistingPhasesFinder.max_revive_per_compartment` for more
        information.
        """
        return self._sizes

    @sizes.setter
    def sizes(self, sizes_new):
        sizes_new = np.array(sizes_new)
        if sizes_new.shape == self._sizes.shape:
            self._sizes = sizes_new
        else:
            self._logger.error(
                f"new sizes with size of {sizes_new.shape} is invalid. It must have the size of {self._sizes.shape}."
            )
            raise ValueError("New sizes must match the size of the old one.")
        self._revive_count_left = (
            self._max_revive_per_compartment * self._num_compartments
        )

    @property
    def phis(self) -> np.ndarray:
        """
        The volume fractions field :math:`\\phi_i^{(m)}`, with the size of
        :math:`N_\\mathrm{c} \\times M`. This property should not be reset in most cases.
        Resetting this property also takes no effect, since :math:`\\phi_i^{(m)}` is not
        considered as the master internal state of the algorithm. Any set values will be
        ignored. If you intend to initialize the system from certain volume fractions,
        please refer to :meth:`reinitialize_from_phis`.
        """

        return self._phis

    @property
    def omegas(self) -> np.ndarray:
        """
        The internal conjugate field :math:`\\omega_i^{(m)}`, with the size of
        :math:`N_\\mathrm{c} \\times M`. This property should not be reset in most cases.
        Resetting this property will change the state of the system, but skip necessary
        reset of other internal states. If you intend to initialize the system from a
        custom conjugate field, please refer to :meth:`reinitialize_from_omegas`.
        """

        return self._omegas

    @property
    def diagnostics(self) -> dict:
        """
        The diagnostics of the most recent call of :meth:`run`. The diagnostics contain
        the convergence status and the original volume fractions before the clustering and
        sorting algorithm is utilized to determine the unique phases.
        """

        return self._diagnostics

    def run(
        self,
        *,
        max_steps: Optional[float] = None,
        tolerance: Optional[float] = None,
        interval: Optional[int] = None,
        progress: Optional[bool] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run instance to find coexisting phases. The keywords arguments can be used to
        temporarily overwrite the provided values during construction of the class. Note
        that this temporary values will not affect the defaults. See class
        :class:`CoexistingPhasesFinder` for other tunable parameters. After each call, the
        diagnostics will be updated, which can be used to, for example, inspecting whether
        the iteration really converges. See :attr:`diagnostics` for more information.

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
            [0]:
                Volume fractions of each phase :math:`J_\\alpha`. 1D array with the size of
                :math:`N_\\mathrm{p}`.
            [1]:
                Volume fractions of components in each phase :math:`\\phi_i^{(\\alpha)}`.
                2D array with the size of :math:`N_\\mathrm{p} \\times N_\\mathrm{c}`.
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

        chis_shifted = self._chis.copy()
        chis_shifted += -self._chis.min() + self._additional_chis_shift

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
        pbar4 = tqdm(**bar_text_args, position=3, desc="Revive Count Left")

        bar_val_func = lambda a: max(0, min(round(-np.log10(max(a, 1e-100)), 1), bar_max))

        for _ in range(steps_tracker):
            # do the inner steps
            (
                max_abs_incomp,
                max_abs_omega_diff,
                max_abs_Js_diff,
                revive_count,
                is_last_step_safe,
            ) = multicomponent_self_consistent_metastep(
                self._phi_means,
                chis_shifted,
                self._sizes,
                omegas=self._omegas,
                Js=self._Js,
                phis=self._phis,
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
                pbar4.n = 1
                pbar1.refresh()
                pbar2.refresh()
                pbar3.refresh()
                pbar4.set_description_str(
                    "{:<20}: {}".format("Revive Count Left", self._revive_count_left)
                )
                pbar4.refresh()

            # check convergence
            if self._convergence_criterion == "standard":
                if (
                    is_last_step_safe
                    and tolerance > max_abs_incomp
                    and tolerance > max_abs_omega_diff
                    and tolerance > max_abs_Js_diff
                ):
                    self._logger.info(
                        f"Composition and volumes reached stationary state after {steps} steps"
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

        # get final result
        final_Js = self._Js.copy()
        final_phis = self._phis.copy()
        revive_compartments_by_copy(final_Js, final_phis, self._kill_threshold, self._rng)

        # store diagnostic output
        self._diagnostics = {
            "steps": steps,
            "max_abs_incomp": max_abs_incomp,
            "max_abs_omega_diff": max_abs_omega_diff,
            "max_abs_js_diff": max_abs_Js_diff,
            "revive_count_left": self._revive_count_left,
            "phis": final_phis,
            "Js": final_Js,
        }

        phases_volumes, phases_compositions = get_clusters(final_Js, final_phis)

        return phases_volumes, phases_compositions


def find_coexisting_phases(
    chis: np.ndarray,
    phi_means: np.ndarray,
    num_compartments: int,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The convenience wrapper for class :class:`CoexistingPhasesFinder`. This function will
    create the class :class:`CoexistingPhasesFinder` internally, and then conduct the
    random initialization, finally use self consistent iterations to find coexisting
    phases. :paramref:`kwargs` is forwarded to :class:`CoexistingPhasesFinder`. See class
    :class:`CoexistingPhasesFinder` for all the possible options.

    Args:
        chis:
            The interaction matrix. 2D array with size of :math:`N_\\mathrm{c} \\times
            N_\\mathrm{c}`. This matrix should be the full :math:`\\chi_{ij}` matrix of
            the system, including the solvent component. Note that the matrix must be
            symmetric, which is not checked but should be guaranteed externally.
        phi_means:
            The average volume fractions :math:`\\bar{\\phi}_i` of all the components of
            the system. 1D array with size of :math:`N_\\mathrm{c}`. Note that the volume
            fraction of the solvent is included as well, therefore the sum of this array
            must be unity, which is not checked by this function and should be guaranteed
            externally.
        num_compartments:
            Number of compartments :math:`M` in the system.
        kwargs:
            See class :class:`CoexistingPhasesFinder` for all the possible options.

    Returns:
        [0]:
            Volume fractions of each phase :math:`J_\\alpha`. 1D array with the size of
            :math:`N_\\mathrm{p}`.
        [1]:
            Volume fractions of components in each phase :math:`\\phi_i^{(\\alpha)}`. 2D
            array with the size of :math:`N_\\mathrm{p} \\times N_\\mathrm{c}`.
    """
    finder = CoexistingPhasesFinder(chis, phi_means, num_compartments, **kwargs)
    return finder.run()
