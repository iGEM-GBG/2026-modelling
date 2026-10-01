"""
Module 4: Reporter Expression (Ste12 -> mRNA -> Reporter maturation)
=======================================================================
Models de-repression of Ste12, transcription of the reporter gene,
translation, and fluorophore maturation -- the final stage of the
biosensor, producing the actual measurable signal.

    Fus3* + Ste12  --k_activate_Ste12-->  Ste12*      (Dig1/Dig2 released)
    Ste12*         --k_deactivate_Ste12--> Ste12

    (basal) + Ste12* --k_transcribe-->  mRNA          (v1: linear, no Hill yet)
    mRNA           --k_degrade_mRNA-->  (degraded)

    mRNA           --k_translate-->     Reporter_immature (dark)
    Reporter_immature --k_mat-->        Reporter_mature   (fluorescent)
    Reporter_mature --k_degrade_mature--> (degraded)

Mechanism note (Ste12 de-repression): in the unstimulated cell, Ste12 is
bound and inhibited by Dig1p/Dig2p. Fus3* (and Kss1*) phosphorylate
Dig1p/Dig2p, which releases them from Ste12p, relieving repression --
source: SGD STE12 locus page (yeastgenome.org/locus/S000001126).

Conservation law (Ste12 only):
    Ste12_total = Ste12 + Ste12*

State vector: y = [Ste12_active, mRNA, Reporter_immature, Reporter_mature]

Output: [Reporter_mature](t) -- the measurable biosensor signal.
No downstream module; this is the end of the pipeline.

Chain: Module 3 [Fus3*](t) --> text file --> interp1d --> input here

MODELING STATUS
----------------
* v1 = linear/mass-action promoter response (Ste12* enters the
  transcription term linearly), NOT a Hill function -- see v2.
* mRNA and both protein pools have NO conservation law (unlike every
  earlier module) -- transcription/translation/degradation are
  birth-death processes bounded only by upstream signal.
* Basal (leaky) transcription is included. Initial conditions are set
  to the analytical basal steady state (Ste12* = 0, only leaky
  transcription active) rather than zero.
* Immature reporter protein has no independent degradation channel --
  it only ever matures.
* Reporter (mature protein) degradation assumes a STABLE, non-degron
  reporter (7 h half-life, Mateus & Avery 2000). With a reporter this
  stable, [Reporter_mature] will NOT plateau within the simulated
  window -- it keeps climbing. Swap k_degrade_mature if a degron tag
  is added.

Authors : iGEM team Gothenburg
Date    : 2026
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ──────────────────────────────────────────────────────────────
# INTERMEDIATE I/O
# ──────────────────────────────────────────────────────────────
INTERMEDIATE_DIR = "../intermediate"
FIGURE_DIR = "../figures"
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

def load_module3_output(L_nM):
    """Load [Fus3*](t) written by module3_mapk.py for a given [L]."""
    path = os.path.join(INTERMEDIATE_DIR, f"module3_output_L{L_nM:.1f}nM.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run modules 1, 2, then 3 first to "
            f"generate this module's input for [L] = {L_nM} nM."
        )
    data = np.loadtxt(path)
    return data[:, 0], data[:, 4]   # t, Fus3_active (column 4 -- see module3 header)

# ──────────────────────────────────────────────────────────────
# PARAMETERS
# ──────────────────────────────────────────────────────────────
# Ste12 pool: SGD median abundance (Ho et al. 2018 unified dataset),
# same method/conversion as Module 3's totals (2668 molecules/cell).
Ste12_total = 105.5   # nM

# Ste12 switch kinetics -- Kofahl & Klipp (2004) Table 2 model values
# (k34, k35), converted min^-1 -> s^-1.
k_activate_Ste12   = 0.3     # nM^-1 s^-1
k_deactivate_Ste12 = 0.167   # s^-1

# Transcription: no specific measured rate exists for this synthetic
# LexA-operator promoter; unchanged from the original order-of-
# magnitude placeholder.
k_transcribe = 5e-4   # s^-1

# k_transcribe_basal: revised from wet-lab fold-change data (thesis
# Table 8, construct T11/TAAR1, n=3, the only statistically supported
# result: control=8.2, treated=12.1, fold=1.476). Solving
# fold = (k_basal + k_transcribe*Ste12_total) / k_basal for k_basal
# gives ~0.111 nM/s -- about 85x larger than the original literature-
# based placeholder (1.3e-3 nM/s, which assumed a generic 20-50x
# pheromone-promoter fold-induction that doesn't hold for this
# specific LexA-operator synthetic promoter). Inherits this result's
# own statistical weakness (single dose, uncorrected p=0.019).
k_transcribe_basal = 0.11    # nM/s

# mRNA degradation: average yeast mRNA half-life ~20 min (genome-wide
# range ~3-90+ min). Source: Wang, Liu, Storey et al. (2002) PNAS
# 99(9):5860-5865. k = ln(2) / (20 min * 60 s/min)
k_degrade_mRNA = 5.78e-4   # s^-1

# Translation: no specific sourced value.
k_translate = 2e-2   # s^-1

# Maturation: GFP variant (yEGFP/GFPmut3 class). Guerra et al. (2022)
# ACS Synth Biol 11:1129-1141 measured sfGFP at 6.9 min half-time in
# yeast but did not test GFPmut3; using 15 min as a middle-of-range
# estimate.
k_mat = 7.70e-4   # s^-1

# Reporter (mature protein) degradation: stability not yet decided;
# defaulting to STABLE/native. Native yEGFP3 half-life ~7 hours.
# Source: Mateus & Avery (2000) Yeast 16(14):1313-1323. If a degron
# tag (e.g. Cln2-PEST) is added later, swap to ~30 min (k ~3.85e-4 s^-1).
k_degrade_mature = 2.75e-5   # s^-1

# ──────────────────────────────────────────────────────────────
# ODE DEFINITION
# ──────────────────────────────────────────────────────────────
def reporter_ode(t, y, k_act12, k_deact12, k_txn, k_txn_basal, k_deg_mRNA,
                  k_tln, k_mat_, k_deg_mat, Ste12_tot, Fus3_interp):
    """
    Reporter expression ODEs (v1 -- see MODELING STATUS above).

    State vector:
        y[0] = Ste12_active
        y[1] = mRNA
        y[2] = Reporter_immature (dark)
        y[3] = Reporter_mature   (fluorescent -- the measured output)
    """
    Ste12a, mRNA, Rep_dark, Rep_mat = y

    Fus3a = max(float(Fus3_interp(t)), 0.0)   # numerical safety clamp

    dSte12a_dt = (k_act12 * Fus3a * (Ste12_tot - Ste12a) - k_deact12 * Ste12a)
    dmRNA_dt   = (k_txn_basal + k_txn * Ste12a - k_deg_mRNA * mRNA)
    dRepD_dt   = (k_tln * mRNA - k_mat_ * Rep_dark)
    dRepM_dt   = (k_mat_ * Rep_dark - k_deg_mat * Rep_mat)

    return [dSte12a_dt, dmRNA_dt, dRepD_dt, dRepM_dt]

# ──────────────────────────────────────────────────────────────
# INITIAL CONDITIONS -- basal steady state
# ──────────────────────────────────────────────────────────────
# Valid in closed form because the promoter response is linear (v1) and
# Ste12* = 0 at basal (no Fus3* before stimulation).
mRNA_basal     = k_transcribe_basal / k_degrade_mRNA
Rep_dark_basal = k_translate * mRNA_basal / k_mat
Rep_mat_basal  = k_translate * mRNA_basal / k_degrade_mature
y0 = [0.0, mRNA_basal, Rep_dark_basal, Rep_mat_basal]

# ──────────────────────────────────────────────────────────────
# SIMULATION
# ──────────────────────────────────────────────────────────────
# LSODA (not RK45): the Kofahl & Klipp-sourced rate constants make
# this system numerically stiff (fast sub-processes alongside the
# hours-long simulation window); RK45 stalls, LSODA auto-switches
# to an implicit stiff method and solves it in milliseconds.
t_start  = 0
t_end    = 6 * 3600     # 6 hours -- extended from Modules 1-3's 3 hours,
                         # since transcription/translation/maturation are
                         # slower processes
n_points = 5000

L_values_nM = [1e5]     # matching Modules 1-3 (100 uM tyramine, wet-lab dose)
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(L_values_nM)))

results_m4 = {}

for L in L_values_nM:
    t_m3, Fus3_m3 = load_module3_output(L)
    Fus3_interp = interp1d(t_m3, Fus3_m3, kind='cubic', fill_value='extrapolate')

    t_eval = np.linspace(t_start, t_end, n_points)
    sol = solve_ivp(
        fun    = reporter_ode,
        t_span = (t_start, t_end),
        y0     = y0,
        t_eval = t_eval,
        args   = (k_activate_Ste12, k_deactivate_Ste12,
                  k_transcribe, k_transcribe_basal, k_degrade_mRNA,
                  k_translate, k_mat, k_degrade_mature,
                  Ste12_total, Fus3_interp),
        method = 'RK45',
        rtol   = 1e-8,
        atol   = 1e-10,
    )
    results_m4[L] = sol

    out_path = os.path.join(INTERMEDIATE_DIR, f"module4_output_L{L:.1f}nM.txt")
    header = (
        "Module 4 output -- Reporter expression (final biosensor signal)\n"
        f"L = {L} nM ; Ste12_total = {Ste12_total} nM ; "
        f"k_mat = {k_mat} 1/s ; k_degrade_mature = {k_degrade_mature} 1/s\n"
        "columns: time_s   Ste12_active_nM   mRNA_nM   "
        "Reporter_immature_nM   Reporter_mature_nM"
    )
    np.savetxt(out_path,
               np.column_stack([sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]]),
               header=header, fmt="%.6e")

L_demo = L_values_nM[0]

# ──────────────────────────────────────────────────────────────
# PLOTTING -- each panel saved as its own PDF
# ──────────────────────────────────────────────────────────────

# ── A: Reporter_mature(t) -- final biosensor signal ───────────
figA, axA = plt.subplots(figsize=(6.5, 5))
for (L, sol), c in zip(results_m4.items(), colors):
    axA.plot(sol.t / 60, sol.y[3], color=c, lw=2, label=f"[L] = {L:.0f} nM")
axA.set_xlabel("Time (min)")
axA.set_ylabel("[Reporter$_{mature}$] (nM)")
axA.set_title("Fluorescent reporter over time\n(final biosensor readout)")
axA.legend(fontsize=9, loc='upper left')
axA.set_xlim(0, t_end / 60)
axA.set_ylim(bottom=0)
figA.tight_layout()
figA.savefig(os.path.join(FIGURE_DIR, "module4_A_reporter_timecourse.pdf"))
plt.close(figA)

# ── B: All four species at the simulated dose ─────────────────
figB, axB = plt.subplots(figsize=(6.5, 5))
sol_demo = results_m4[L_demo]
t_min = sol_demo.t / 60
axB.plot(t_min, sol_demo.y[0], lw=2, color='steelblue',  label='Ste12*')
axB.plot(t_min, sol_demo.y[1], lw=2, color='darkorange', label='mRNA')
axB.plot(t_min, sol_demo.y[2], lw=2, color='gray',       label='Reporter (dark)')
axB.plot(t_min, sol_demo.y[3], lw=2, color='forestgreen', label='Reporter (mature)')
axB.set_xlabel("Time (min)")
axB.set_ylabel("Concentration (nM)")
axB.set_title(f"All four species\n([L] = {L_demo:.0f} nM, wet-lab dose)")
axB.legend(fontsize=9)
axB.set_xlim(0, t_end / 60)
figB.tight_layout()
figB.savefig(os.path.join(FIGURE_DIR, "module4_B_all_species.pdf"))
plt.close(figB)

# ── C: Reporter stability sensitivity (stable vs destabilized) ──
figC, axC = plt.subplots(figsize=(6.5, 5))
t_m3, Fus3_m3 = load_module3_output(L_demo)
Fus3_interp_fixed = interp1d(t_m3, Fus3_m3, kind='cubic', fill_value='extrapolate')
t_eval = np.linspace(t_start, t_end, n_points)

k_deg_variants = {
    'stable (7 h t\u00bd, this script)':        k_degrade_mature,
    'destabilized (30 min t\u00bd, Cln2-PEST)': np.log(2) / (30 * 60),
    'fast degron (~7 min t\u00bd)':             np.log(2) / (7 * 60),
}
variant_colors = ['#2ca02c', '#ff7f0e', '#d62728']

for (label, kdeg), c in zip(k_deg_variants.items(), variant_colors):
    rep_mat_basal_v = k_translate * mRNA_basal / kdeg
    y0_v = [0.0, mRNA_basal, Rep_dark_basal, rep_mat_basal_v]
    sol_v = solve_ivp(
        reporter_ode, (t_start, t_end), y0_v, t_eval=t_eval,
        args=(k_activate_Ste12, k_deactivate_Ste12,
              k_transcribe, k_transcribe_basal, k_degrade_mRNA,
              k_translate, k_mat, kdeg,
              Ste12_total, Fus3_interp_fixed),
        method='LSODA', rtol=1e-8, atol=1e-10
    )
    axC.plot(sol_v.t / 60, sol_v.y[3], lw=2, color=c, label=label)

axC.set_xlabel("Time (min)")
axC.set_ylabel("[Reporter$_{mature}$] (nM)")
axC.set_title(f"Reporter stability matters\n[L] = {L_demo:.0f} nM")
axC.legend(fontsize=9)
axC.set_xlim(0, t_end / 60)
figC.tight_layout()
figC.savefig(os.path.join(FIGURE_DIR, "module4_C_stability_sensitivity.pdf"))
plt.close(figC)

print("Figures saved: module4_{A_reporter_timecourse,B_all_species,"
      "C_stability_sensitivity}.pdf")

# ──────────────────────────────────────────────────────────────
# SIGNAL SUMMARY
# ──────────────────────────────────────────────────────────────
print(f"\n-- Signal at t = {t_end/3600:.0f}h (NOT steady state -- stable reporter) --")
print(f"{'[L] (nM)':>10} {'[Ste12*]':>10} {'[mRNA]':>10} "
      f"{'[Rep_dark]':>12} {'[Rep_mature]':>13}")
print("-" * 60)
for L, sol in results_m4.items():
    s12 = sol.y[0][-1]
    m   = sol.y[1][-1]
    rd  = sol.y[2][-1]
    rm  = sol.y[3][-1]
    print(f"{L:>10.1f} {s12:>10.3f} {m:>10.3f} {rd:>12.3f} {rm:>13.3f}")

print("\n-- Basal (t=0) initial conditions ----------------------------------")
print(f"  mRNA_basal          = {mRNA_basal:.4f} nM")
print(f"  Reporter_dark_basal = {Rep_dark_basal:.4f} nM")
print(f"  Reporter_mature_basal (LOD floor) = {Rep_mat_basal:.4f} nM")
