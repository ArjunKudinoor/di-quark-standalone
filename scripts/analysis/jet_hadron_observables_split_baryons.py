#!/usr/bin/env python3
"""Fig. 3 p/n selection, following the original and pi0 event processors.

Only PDG 2212 and 2112 are eligible (not antiparticles). Kinematics, PYTHIA
cards, final-hadron collection, zLC and transverse observables are reused.
No existing module is patched and no pion parent reconstruction is used.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import jet_hadron_observables_split_pi_pm as original
from jet_hadron_observables_split_pi_pm import (
    JH, FLIP_Z_PTREL, ZLC_MAX, build_LT, flip_z, is_hadron, p3,
    extract_beams, row_p4, dis_xy_q_within_selection, final_colored_partons,
    color_balance_ok, flipped_lab_beam_arrays_from_event_row,
    kinematics_lab_to_breit_lt_from_flip_lab_beams,
    collider_forward_pion_pseudorapidity,
    pion_lab_eta_native_beam_from_breit as baryon_lab_eta_native_beam_from_breit,
    compute_observables_for_pion as compute_observables_for_baryon,
    run_pythia_reinject_collect,
)

BARYON_PDGS = (2212, 2112)


def build_pythia_reinjector(**kwargs):
    """Original cards and RNG; check rather than change neutron decay settings."""
    pythia = original.build_pythia_reinjector(**kwargs)
    if pythia.particleData.mayDecay(2112):
        raise RuntimeError(
            'Neutron mayDecay is enabled: final-state neutron selection would '
            'not match the inspected PYTHIA 8.315 convention. No settings changed.'
        )
    return pythia


def discard_baryons_outside_collider_eta(hadrons, LT_lab_to_breit, eta_min, eta_max):
    """Same strict proton-forward lab eta filter, applied to p and n alike."""
    LT = np.asarray(LT_lab_to_breit, dtype=np.float64)
    try:
        LT_inv = np.linalg.inv(LT)
    except np.linalg.LinAlgError:
        LT_inv = np.linalg.pinv(LT)
    out = []
    for pid, p4 in hadrons:
        if int(pid) in BARYON_PDGS and is_hadron(int(pid)):
            p_beam_lab = flip_z(LT_inv @ np.asarray(p4, dtype=float), FLIP_Z_PTREL)
            try:
                eta = collider_forward_pion_pseudorapidity(p_beam_lab)
            except ValueError:
                continue
            if not math.isfinite(eta) or not (eta_min < eta < eta_max):
                continue
        out.append((int(pid), np.asarray(p4, dtype=float).copy()))
    return out


def best_leading_baryon_breit(hadrons, *, breit_hemisphere='target'):
    """One maximum-Breit-energy p OR n, preserving original tie/order rules."""
    if breit_hemisphere not in ('target', 'current'):
        raise ValueError('breit_hemisphere must be target or current')
    best_e, best_pid, best_p4 = -1., None, None
    for pid, p4 in hadrons:
        if int(pid) not in BARYON_PDGS or not is_hadron(int(pid)):
            continue
        p4 = np.asarray(p4, dtype=float)
        energy, pz = float(p4[0]), float(p4[3])
        if (breit_hemisphere == 'target' and pz <= 0) or (breit_hemisphere == 'current' and pz >= 0):
            continue
        if energy > best_e:
            best_e, best_pid, best_p4 = energy, int(pid), p4.copy()
    return best_pid, best_p4


def process_event_dataframe(
    df, struck_idx, sample, csv_label, csv_momenta_breit=False, pythia_reuse=None,
    *, event_row=None, baryon_collider_eta_acceptance=(2., 3.5),
    baryon_breit_hemisphere='target', pythia_e_beam_gev=5., pythia_p_beam_gev=41.,
    dis_x_exclusive_floor=.05, dis_q_open_interval_gev=(4., 5.),
    pythia_parton_level_isr=True, pythia_parton_level_fsr=True,
):
    """One observable/failure row per event, with the original cut order.

    DIS -> color -> rehadronization -> lab eta -> maximum E in target Breit
    hemisphere -> original zLC/phi calculation. No fallback if the winner fails.
    The runner has no additional hadron/jet pT cuts, as in run_fig3_pi0.sh.
    """
    df = df.sort_values('particle_index').reset_index(drop=True)
    base = dict(event_id=int(df['event_id'].iloc[0]), sample=sample, csv=csv_label)

    def fail(reason, **extra):
        return [{**base, 'baryon': '', 'pdg_id': 0, 'ok': False, 'reason': reason, **extra}]

    e_in, e_sc, p_in, beam_msg = extract_beams(df)
    if e_in is None:
        return fail(beam_msg)
    k_row = df[df['particle_index'] == struck_idx]
    if len(k_row) != 1:
        return fail('k_out_row')
    k_out = row_p4(k_row.iloc[0])
    if csv_momenta_breit:
        e_in_ev, e_sc_ev, p_in_ev, k_out_ev = e_in, e_sc, p_in, k_out
    else:
        e_in_ev, e_sc_ev, p_in_ev, k_out_ev = (
            flip_z(v, FLIP_Z_PTREL) for v in (e_in, e_sc, p_in, k_out)
        )
    qmu = e_in_ev - e_sc_ev
    Q2 = -(qmu[0]*qmu[0] - qmu[1]*qmu[1] - qmu[2]*qmu[2] - qmu[3]*qmu[3])
    if Q2 <= 0:
        return fail('Q2')
    Q = float(np.sqrt(Q2))
    qT = float(np.hypot(qmu[1], qmu[2]))
    p_dot_q = float(p_in_ev[0]*qmu[0] - p_in_ev[1]*qmu[1]
                    - p_in_ev[2]*qmu[2] - p_in_ev[3]*qmu[3])
    if p_dot_q == 0:
        return fail('pdotq')
    x = Q2 / (2. * p_dot_q)
    if not dis_xy_q_within_selection(x, Q, x_exclusive_floor=dis_x_exclusive_floor,
                                     q_open_interval_gev=dis_q_open_interval_gev):
        return fail('xQ_window')
    Ee, Ep = float(e_in_ev[0]), float(p_in_ev[0])
    S = 4. * Ee * Ep
    y = Q2 / (S * x)
    phiq = float(np.arctan2(qmu[2], qmu[1]))
    LT = np.eye(4) if csv_momenta_breit else build_LT(Ee, Ep, qmu, x, y, qT, phiq, S)
    if LT is None:
        return fail('build_LT')
    partons = final_colored_partons(df)
    color_ok, color_msg = color_balance_ok(partons)
    if not color_ok:
        return fail('color', color_message=color_msg)
    if pythia_reuse is None:
        pythia_reuse = build_pythia_reinjector(
            e_beam_gev=pythia_e_beam_gev, p_beam_gev=pythia_p_beam_gev,
            parton_level_isr=pythia_parton_level_isr, parton_level_fsr=pythia_parton_level_fsr,
        )
    py_ok, py_error, hadrons = run_pythia_reinject_collect(pythia_reuse, partons, LT, csv_momenta_breit)
    if not py_ok:
        return fail('pythia', pythia_error=py_error)
    if baryon_collider_eta_acceptance is not None:
        eta_min, eta_max = map(float, baryon_collider_eta_acceptance)
        if csv_momenta_breit:
            if event_row is None:
                return fail('event_row_missing_for_eta')
            beams = flipped_lab_beam_arrays_from_event_row(event_row)
            if beams is None:
                return fail('missing_lab_beams_for_eta')
            LT_eta, eta_error = kinematics_lab_to_breit_lt_from_flip_lab_beams(
                *beams, x_exclusive_floor=dis_x_exclusive_floor,
                q_open_interval_gev=dis_q_open_interval_gev,
            )
            if LT_eta is None:
                return fail(eta_error or 'build_LT_eta')
        else:
            LT_eta = LT
        hadrons = discard_baryons_outside_collider_eta(hadrons, LT_eta, eta_min, eta_max)
    boost = lambda v: LT @ np.asarray(v, dtype=float)
    p_breit = boost(p_in_ev)
    if float(p_breit[0] + p_breit[3]) <= 0:
        return fail('P_plus')
    # The original outgoing struck-quark proxy, never a reconstructed jet.
    pT_jet = JH._pT_vec_breit(boost(k_out_ev))
    p_rem_breit = boost(p_in_ev - (k_out_ev - qmu))
    if float(np.linalg.norm(p3(p_rem_breit))) <= 0:
        return fail('remnant_pt')
    q_breit = boost(qmu)
    pid, baryon = best_leading_baryon_breit(hadrons, breit_hemisphere=baryon_breit_hemisphere)
    row = dict(**base, baryon='proton' if pid == 2212 else 'neutron' if pid == 2112 else '',
               pdg_id=pid if pid is not None else 0, Q2=float(Q2), xB=float(x),
               color_ok=True, pythia_ok=True, n_final_hadrons=len(hadrons), baryon_lab_eta=float('nan'))
    if baryon is None:
        row.update(ok=False, reason='no_baryon_candidate')
        return [row]
    row['baryon_breit_energy_GeV'] = float(baryon[0])
    row['baryon_breit_pz_GeV'] = float(baryon[3])
    eta = baryon_lab_eta_native_beam_from_breit(
        baryon, csv_momenta_breit=csv_momenta_breit, event_row=event_row,
        LT_breit_lab_to_process=LT, x_exclusive_floor=dis_x_exclusive_floor,
        q_open_interval_gev=dis_q_open_interval_gev,
    )
    if eta is not None:
        row['baryon_lab_eta'] = float(eta)
    row['baryon_pht_GeV'] = float(np.linalg.norm(JH._pT_vec_breit(baryon)))
    row['jet_pjt_GeV'] = float(np.linalg.norm(pT_jet))
    phi, diff, pbar, sum_mag, zLC, message = compute_observables_for_baryon(baryon, pT_jet, p_breit, q_breit)
    if message != 'ok':
        row.update(ok=False, reason=message, zLC=zLC)
        return [row]
    row.update(ok=True, reason='ok', phi_hJ=float(phi) if phi is not None else float('nan'),
               delta_P_hJ_GeV=diff, Pbar_t_GeV=pbar, sum_mag_GeV=sum_mag, zLC=zLC)
    return [row]
