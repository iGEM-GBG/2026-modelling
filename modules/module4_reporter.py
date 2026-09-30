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
source: SGD STE12 locus page (yeastgenome.org/locus/S000001126),
"Dig1p and Dig2p directly inhibit the transcriptional activity of
Ste12p... the inhibitory functions of Dig1p and Dig2p are relieved by
phosphorylation through the mating-specific MAPKs Fus3p and Kss1p."

Conservation law (Ste12 only -- see MODELING STATUS):
    Ste12_total = Ste12 + Ste12*

State vector: y = [Ste12_active, mRNA, Reporter_immature, Reporter_mature]

Output: [Reporter_mature](t) -- the measurable biosensor signal.
No downstream module; this is the end of the pipeline.

Chain: Module 3 [Fus3*](t) --> text file --> interp1d --> input here

MODELING STATUS (read before trusting the dynamics)
----------------------------------------------------
* v1 = linear/mass-action promoter response (Ste12* enters the
  transcription term linearly), NOT a Hill function. You noted
  multiple PREs on the reporter promoter would argue for cooperative
  (Hill-type) kinetics -- that's deferred to v2, same as Module 3's
  planned move from mass action to saturating kinetics.
* mRNA and both protein pools have NO conservation law (unlike every
  earlier module) -- transcription/translation/degradation are
  birth-death processes bounded only by upstream signal, not switches
  between two forms of a fixed pool.
* Basal (leaky) transcription is included. Initial conditions are set
  to the analytical basal steady state (Ste12* = 0, only leaky
  transcription active) rather than zero, since a real cell has been
  sitting at that basal state long before ligand exposure. NOTE: this
  closed-form basal steady state is only valid because the promoter
  response is linear (v1). If Hill kinetics are added in v2, replace
  this with a numerical pre-equilibration run instead.
* Immature reporter protein has no independent degradation channel --
  it only ever matures (by your choice).
* Reporter (mature protein) degradation assumes a STABLE, non-degron
  reporter (7 h half-life, Mateus & Avery 2000 -- see parameters).
  This has a real consequence, not just a caveat: with a reporter this
  stable, [Reporter_mature] will NOT plateau within a few hours -- it
  keeps climbing. This is exactly why Mateus & Avery built a
  destabilized GFP in the first place ("[GFP's] stability makes it
  unsuitable for monitoring dynamic changes in gene expression").
  If/when you decide on reporter stability, swap k_degrade_mature.
* All rate constants for Ste12 activation/deactivation, transcription
  (basal and induced), and translation are PLACEHOLDERS -- order of
  magnitude only. See each parameter's comment for what little
  grounding exists; most don't have a clean literature anchor and are
  flagged as such rather than dressed up with false precision.

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
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

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
#
# --- Ste12 pool and switch kinetics ------------------------------------
# Ste12: median abundance 2668 +/- 678 molecules/cell (SGD unified
# dataset, same method/conversion as Module 3's totals).
#   2668 * 0.03954 ~ 105.5 nM
Ste12_total = 105.5   # nM  -- FINAL (SGD median abundance)

# *** PLACEHOLDERS, no clean literature anchor for Dig1/2 release kinetics ***
# magnitude-matched to Module 3's rate constants for pipeline consistency.
k_activate_Ste12   = 0.3   # nM^-1 s^-1  -- Kofahl and Klipp model value FINAL
k_deactivate_Ste12 = 0.167   # s^-1        -- Kofahl and Klipp model value FINAL

# --- Transcription -------------------------------------------------------
# k_transcribe: *** PLACEHOLDER *** -- no specific measured rate for a
# synthetic pheromone-responsive promoter in this construct; unchanged,
# no data available to revise this specific value.
k_transcribe       = 5e-4    # s^-1        -- PLACEHOLDER

# k_transcribe_basal: REVISED from wet-lab fold-change data (thesis
# Table 8, construct T11/TAAR1, n=3, the only statistically supported
# result: control=8.2, treated=12.1, fold=1.476). Solving
# fold = (k_basal + k_transcribe*Ste12_total) / k_basal for k_basal
# gives ~0.111 nM/s -- about 85x LARGER than the original literature-
# based placeholder (1.3e-3 nM/s, which assumed a generic 20-50x
# pheromone-promoter fold-induction that doesn't hold for this specific
# LexA-operator synthetic promoter). Inherits this result's own
# statistical weakness (single dose, uncorrected p=0.019 -- see
# Open_Issues). Superior to the old placeholder because it's grounded
# in this construct's actual behavior rather than a cross-species
# generic assumption.
k_transcribe_basal = 0.11    # nM/s        -- REVISED (Table 8, T11/TAAR1)

# mRNA degradation: average yeast mRNA half-life ~20 min (genome-wide
# range ~3-90+ min, no strong correlation with length/function).
# Source: Wang, Liu, Storey et al. (2002) PNAS 99(9):5860-5865,
# "Precision and functional specificity in mRNA decay" -- half-lives
# ranging from ~3 min to >90 min genome-wide; Herrick, Delaney &
# Jacobson (1990) Mol Gen Genomics reports an average of ~22 min.
# k = ln(2) / (20 min * 60 s/min)
k_degrade_mRNA = 5.78e-4   # s^-1  -- typical value (Wang et al. 2002 PNAS)

# --- Translation -----------------------------------------------------------
# *** PLACEHOLDER *** -- no specific sourced value.
k_translate = 2e-2   # s^-1  -- PLACEHOLDER

# --- Maturation --------------------------------------------------------
# Reporter: GFP variant (yEGFP/GFPmut3 class), per your input.
# I could NOT find an in-yeast measurement specific to GFPmut3 through
# search -- Guerra, Vuillemenot, Rae, Ladyhina & Milias-Argeitis (2022)
# ACS Synth Biol 11:1129-1141 systematically measured 12 FPs in budding
# yeast and found avGFP-derived variants mature fast via one-step
# kinetics (sfGFP: 6.9 min half-time, 95% CI [5, 10.5]), but did not
# test plain GFPmut3. Using 15 min as a middle-of-range placeholder
# (within the 15-30 min range you indicated) -- replace with a direct
# measurement or the Guerra et al. supplementary data if your exact
# variant is closer to one they tested.
k_mat = 7.70e-4   # s^-1  -- PLACEHOLDER (15 min half-time, see note above)

# --- Reporter (mature protein) degradation --------------------------------
# Stability not yet decided (your input) -- defaulting to STABLE/native,
# since that's the common case absent an explicit degron tag.
# Native (non-destabilized) yEGFP3 half-life in yeast ~ 7 hours.
# Source: Mateus & Avery (2000) Yeast 16(14):1313-1323, "Destabilized
# green fluorescent protein for monitoring dynamic changes in yeast
# gene expression with flow cytometry" -- cited directly (with the 7h
# figure) by the 2009 iGEM DTU-Denmark team's model write-up:
# "fusion of GFP and a PEST degradation signal from... Cln2, which has
# been demonstrated to reduce the half-life from 7 hours to 30 minutes."
# IF you later add a degron tag (e.g. Cln2-PEST), replace this with
# ~30 min half-life instead (k ~ 3.85e-4 s^-1).
# k = ln(2) / (7 h * 3600 s/h)
k_degrade_mature = 2.75e-5   # s^-1  -- PLACEHOLDER default (stable GFP assumed)

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
# INITIAL CONDITIONS -- basal steady state (see MODELING STATUS)
# ──────────────────────────────────────────────────────────────
# Valid in closed form because the promoter response is linear (v1) and
# Ste12* = 0 at basal (no Fus3* before stimulation, no basal leak on
# the Ste12 switch itself -- only the promoter leaks).
mRNA_basal     = k_transcribe_basal / k_degrade_mRNA
Rep_dark_basal = k_translate * mRNA_basal / k_mat
Rep_mat_basal  = k_translate * mRNA_basal / k_degrade_mature
y0 = [0.0, mRNA_basal, Rep_dark_basal, Rep_mat_basal]

# ──────────────────────────────────────────────────────────────
# SIMULATION
# ──────────────────────────────────────────────────────────────
t_start  = 0
t_end    = 6 * 3600     # 4 hours -- extended from Modules 1-3's 3 hours,
                         # since transcription/translation/maturation are
                         # slower processes (per your input)
n_points = 5000

L_values_nM = [1e5]
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

# ──────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)

# ── Panel A: Reporter_mature(t) -- final biosensor signal ────
ax1 = fig.add_subplot(gs[0, 0])
for (L, sol), c in zip(results_m4.items(), colors):
    ax1.plot(sol.t / 60, sol.y[3], color=c, lw=2, label=f"[L] = {L} nM")
ax1.set_xlabel("Time (min)")
ax1.set_ylabel("[Reporter$_{mature}$] (nM)")
ax1.set_title("A.  Fluorescent reporter over time\n(final biosensor readout)")
ax1.legend(fontsize=7, loc='upper left')
ax1.set_xlim(0, t_end / 60)
ax1.set_ylim(bottom=0)

# ── Panel B: All four species for [L] = 5 nM = K_D ───────────
ax2 = fig.add_subplot(gs[0, 1])
L_demo = L_values_nM[0]
sol_demo = results_m4[L_demo]
t_min = sol_demo.t / 60
ax2.plot(t_min, sol_demo.y[0], lw=2, color='steelblue',  label='Ste12*')
ax2.plot(t_min, sol_demo.y[1], lw=2, color='darkorange', label='mRNA')
ax2.plot(t_min, sol_demo.y[2], lw=2, color='gray',       label='Reporter (dark)')
ax2.plot(t_min, sol_demo.y[3], lw=2, color='forestgreen', label='Reporter (mature)')
ax2.set_xlabel("Time (min)")
ax2.set_ylabel("Concentration (nM)")
ax2.set_title(f"B.  All four species\n([L] = {L_demo} nM = $K_D$)")
ax2.legend(fontsize=8)
ax2.set_xlim(0, t_end / 60)

# ── Panel C: [Reporter_mature] at t = t_end vs [L] (dose-response) ──
# NOTE: with a stable reporter this is NOT a true steady state (see
# MODELING STATUS) -- it's the signal at a fixed assay time, which is
# how a real plate-reader/flow measurement would actually be read out.
ax3 = fig.add_subplot(gs[1, 0])
Rep_at_tend = [results_m4[L].y[3][-1] for L in L_values_nM]
ax3.semilogx(L_values_nM, Rep_at_tend, 'o-', color='forestgreen', lw=2, ms=8)
ax3.axvline(5.0, color='crimson', ls='--', lw=1.5, label='$K_D$ = 5 nM')
ax3.set_xlabel("[L] (nM, log scale)")
ax3.set_ylabel(f"[Reporter$_{{mature}}$] at t={t_end/3600:.0f}h (nM)")
ax3.set_title("C.  Reporter signal vs [L]\n(NOT steady state -- see script notes)")
ax3.legend(fontsize=9)
#ax3.set_ylim(bottom=0)

# ── Panel D: k_degrade_mature sensitivity (stable vs destabilized) ──
ax4 = fig.add_subplot(gs[1, 1])
L_fixed = 5.0
t_m3, Fus3_m3 = load_module3_output(L_fixed)
Fus3_interp_fixed = interp1d(t_m3, Fus3_m3, kind='cubic', fill_value='extrapolate')
t_eval = np.linspace(t_start, t_end, n_points)

k_deg_variants = {
    'stable (7 h t\u00bd, this script)': k_degrade_mature,
    'destabilized (30 min t\u00bd, Cln2-PEST)': np.log(2) / (30 * 60),
    'fast degron (~7 min t\u00bd)': np.log(2) / (7 * 60),
}
variant_colors = ['#2ca02c', '#ff7f0e', '#d62728']

for (label, kdeg), c in zip(k_deg_variants.items(), variant_colors):
    # recompute matching basal steady state for a fair comparison
    rep_mat_basal_v = k_translate * mRNA_basal / kdeg
    y0_v = [0.0, mRNA_basal, Rep_dark_basal, rep_mat_basal_v]
    sol_v = solve_ivp(
        reporter_ode, (t_start, t_end), y0_v, t_eval=t_eval,
        args=(k_activate_Ste12, k_deactivate_Ste12,
              k_transcribe, k_transcribe_basal, k_degrade_mRNA,
              k_translate, k_mat, kdeg,
              Ste12_total, Fus3_interp_fixed),
        method='RK45', rtol=1e-8, atol=1e-10
    )
    ax4.plot(sol_v.t / 60, sol_v.y[3], lw=2, color=c, label=label)

ax4.set_xlabel("Time (min)")
ax4.set_ylabel("[Reporter$_{mature}$] (nM)")
ax4.set_title("D.  Reporter stability matters\n"
              f"[L] = {L_fixed} nM = $K_D$")
ax4.legend(fontsize=7)
ax4.set_xlim(0, t_end / 60)

fig.suptitle(
    "Module 4 -- Reporter Expression (linear v1, basal leak, stable-GFP default)\n"
    "Ste12_total from SGD median abundance [PLACEHOLDER]  |  "
    "k_mat, k_degrade_mRNA, k_degrade_mature partially sourced (see script)  |  "
    "transcription/translation rates PLACEHOLDER",
    fontsize=9, y=1.01
)

plt.savefig("../figures/module4_reporter.png", dpi=150, bbox_inches='tight')
plt.close()
print("Figure saved.")

# ──────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ──────────────────────────────────────────────────────────────
print(f"\n── Signal at t = {t_end/3600:.0f}h (NOT steady state -- stable reporter) ──────")
print(f"{'[L] (nM)':>10} {'[Ste12*]':>10} {'[mRNA]':>10} "
      f"{'[Rep_dark]':>12} {'[Rep_mature]':>13}")
print("-" * 60)
for L, sol in results_m4.items():
    s12 = sol.y[0][-1]
    m   = sol.y[1][-1]
    rd  = sol.y[2][-1]
    rm  = sol.y[3][-1]
    print(f"{L:>10.1f} {s12:>10.3f} {m:>10.3f} {rd:>12.3f} {rm:>13.3f}")

print("\n── Basal (t=0) initial conditions ─────────────────────────────────")
print(f"  mRNA_basal          = {mRNA_basal:.4f} nM")
print(f"  Reporter_dark_basal = {Rep_dark_basal:.4f} nM")
print(f"  Reporter_mature_basal (LOD floor) = {Rep_mat_basal:.4f} nM")

print("\n── Parameter status ──────────────────────────────────────────────")
print(f"  Ste12_total       = {Ste12_total} nM      <- PLACEHOLDER (SGD median abundance)")
print("  k_activate/deactivate_Ste12                <- PLACEHOLDER, no clean source")
print("  k_transcribe, k_transcribe_basal            <- PLACEHOLDER, ratio loosely")
print("                                                  informed by typical fold-induction")
print(f"  k_degrade_mRNA    = {k_degrade_mRNA:.2e} 1/s <- ~20 min half-life (Wang et al. 2002 PNAS)")
print("  k_translate                                 <- PLACEHOLDER, no source")
print(f"  k_mat             = {k_mat:.2e} 1/s <- 15 min placeholder (Guerra et al. 2022")
print("                                                  measured sfGFP at 6.9 min; GFPmut3 not")
print("                                                  directly tested in that study)")
print(f"  k_degrade_mature  = {k_degrade_mature:.2e} 1/s <- stable-GFP default, 7h half-life")
print("                                                  (Mateus & Avery 2000). SWAP if you add")
print("                                                  a degron tag (e.g. ~30 min, same paper)")
