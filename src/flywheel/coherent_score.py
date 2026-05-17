"""Higher-mode coherent-score wrapper used by the low-latency workflow."""

from __future__ import annotations

import numpy as np
import os
import sys


def _require_cogwheel():
    cogwheel_path = os.environ.get("FLYWHEEL_COGWHEEL_PATH")
    if cogwheel_path:
        sys.path.insert(0, cogwheel_path)
    try:
        import cogwheel  # noqa: PLC0415
        from cogwheel.likelihood.marginalization.base import (  # noqa: PLC0415
            BaseCoherentScoreHM,
            MarginalizationInfoHM,
        )
    except ImportError as exc:
        raise ImportError(
            "flywheel's coherent-score calculation requires cogwheel. "
            "Install cogwheel or set FLYWHEEL_COGWHEEL_PATH to a cogwheel checkout "
            "as described in README.md. The import failed because of: "
            f"{exc}"
        ) from exc
    return cogwheel, BaseCoherentScoreHM, MarginalizationInfoHM


cogwheel, BaseCoherentScoreHM, MarginalizationInfoHM = _require_cogwheel()


class SearchCoherentScoreHMAS(BaseCoherentScoreHM):
    """Marginalized coherent score for aligned-spin 22+33+44 waveforms."""

    M_ARR = np.array([2, 3, 4])

    def __init__(
        self,
        *,
        sky_dict,
        m_arr=M_ARR,
        lookup_table=None,
        log2n_qmc: int = 11,
        nphi: int = 128,
        seed: int | None = 0,
        beta_temperature: float = 0.1,
        n_qmc_sequences: int = 128,
        min_n_effective: int = 50,
        max_log2n_qmc: int = 15,
    ):
        if not np.array_equal(m_arr, self.M_ARR):
            raise ValueError(f"`m_arr` must be {self.M_ARR} in this class.")
        super().__init__(
            m_arr=self.M_ARR,
            sky_dict=sky_dict,
            lookup_table=lookup_table,
            log2n_qmc=log2n_qmc,
            nphi=nphi,
            seed=seed,
            beta_temperature=beta_temperature,
            n_qmc_sequences=n_qmc_sequences,
            min_n_effective=min_n_effective,
            max_log2n_qmc=max_log2n_qmc,
        )

    @property
    def _qmc_range_dic(self):
        return super()._qmc_range_dic | {"cosiota": (-1, 1)}

    def _create_qmc_sequence(self):
        qmc_sequence = super()._create_qmc_sequence()
        siniota = np.sin(np.arccos(qmc_sequence["cosiota"]))
        qmc_sequence["response"] = np.einsum(
            "Pq,qPp,qm->qpm",
            (
                (1 + qmc_sequence["cosiota"] ** 2) / 2,
                -1j * qmc_sequence["cosiota"],
            ),
            qmc_sequence["rot_psi"],
            np.power.outer(siniota, np.arange(3)),
        )
        return qmc_sequence

    def get_marginalization_info(
        self,
        dh_mtd,
        hh_md,
        times,
        incoherent_lnprob_td,
        mode_ratios_qm,
    ):
        """Integrate over sky location, distance, inclination and phases."""
        self.sky_dict.set_generators()

        dh_mtd, _ = self.sky_dict.resample_timeseries(dh_mtd, times, axis=-2)
        t_arrival_lnprob, times = self.sky_dict.resample_timeseries(
            incoherent_lnprob_td.T, times, axis=-1
        )

        self.sky_dict.apply_tdet_prior(t_arrival_lnprob)
        t_arrival_prob = cogwheel.utils.exp_normalize(t_arrival_lnprob, axis=1)
        return self._get_marginalization_info(
            dh_mtd, hh_md, times, t_arrival_prob, mode_ratios_qm=mode_ratios_qm
        )

    def _get_marginalization_info_chunk(
        self,
        dh_mtd,
        hh_md,
        times,
        t_arrival_prob,
        i_chunk,
        mode_ratios_qm,
    ):
        q_inds = self._qmc_ind_chunks[i_chunk]
        n_qmc = len(q_inds)
        tdet_inds = self._get_tdet_inds(t_arrival_prob, q_inds)

        sky_inds, sky_prior, physical_mask = self.sky_dict.get_sky_inds_and_prior(
            tdet_inds[1:] - tdet_inds[0]
        )

        q_inds = q_inds[physical_mask]
        tdet_inds = tdet_inds[:, physical_mask]
        if not any(physical_mask):
            return MarginalizationInfoHM(
                qmc_sequence_id=self._current_qmc_sequence_id,
                ln_numerators=np.array([]),
                q_inds=np.array([], int),
                o_inds=np.array([], int),
                sky_inds=np.array([], int),
                t_first_det=np.array([]),
                d_h=np.array([]),
                h_h=np.array([]),
                tdet_inds=tdet_inds,
                proposals_n_qmc=[n_qmc],
                proposals=[t_arrival_prob],
                flip_psi=np.array([], bool),
            )

        t_first_det = times[tdet_inds[0]] + self._qmc_sequence["t_fine"][q_inds]
        dh_qo, hh_qo = self._get_dh_hh_qo(
            sky_inds, q_inds, t_first_det, times, dh_mtd, hh_md, mode_ratios_qm
        )
        ln_numerators, important, flip_psi = (
            self._get_lnnumerators_important_flippsi(dh_qo, hh_qo, sky_prior)
        )

        q_inds = q_inds[important[0]]
        sky_inds = sky_inds[important[0]]
        t_first_det = t_first_det[important[0]]
        tdet_inds = tdet_inds[:, important[0]]
        return MarginalizationInfoHM(
            qmc_sequence_id=self._current_qmc_sequence_id,
            ln_numerators=ln_numerators,
            q_inds=q_inds,
            o_inds=important[1],
            sky_inds=sky_inds,
            t_first_det=t_first_det,
            d_h=dh_qo[important],
            h_h=hh_qo[important],
            tdet_inds=tdet_inds,
            proposals_n_qmc=[n_qmc],
            proposals=[t_arrival_prob],
            flip_psi=flip_psi,
        )

    def lnlike_marginalized(
        self,
        dh_mtd,
        hh_md,
        times,
        incoherent_lnprob_td,
        mode_ratios_qm,
    ):
        marg_info = self.get_marginalization_info(
            dh_mtd,
            hh_md,
            times,
            incoherent_lnprob_td,
            mode_ratios_qm=mode_ratios_qm,
        )
        return marg_info.lnl_marginalized

    def _get_dh_hh_qo(
        self,
        sky_inds,
        q_inds,
        t_first_det,
        times,
        dh_mtd,
        hh_md,
        mode_ratios_qm,
    ):
        t_det = np.vstack((t_first_det, t_first_det + self.sky_dict.delays[:, sky_inds]))
        dh_dmq = np.array(
            [
                self._interp_locally(times, dh_mtd[..., i_det], t_det[i_det])
                for i_det in range(len(self.sky_dict.detector_names))
            ]
        )

        factor_qdm = (
            self.sky_dict.fplus_fcross_0[sky_inds] @ self._qmc_sequence["response"][q_inds]
        )
        factor_qdm[..., 1:] *= mode_ratios_qm[q_inds, np.newaxis, :]

        dh_qm = np.einsum("dmq,qdm->qm", dh_dmq, factor_qdm.conj())
        hh_qm = np.einsum(
            "md,qdm,qdm->qm",
            hh_md,
            factor_qdm[..., self.m_inds],
            factor_qdm.conj()[..., self.mprime_inds],
        )
        dh_qo = cogwheel.utils.real_matmul(dh_qm, self._dh_phasor)
        hh_qo = cogwheel.utils.real_matmul(hh_qm, self._hh_phasor)
        return dh_qo, hh_qo
