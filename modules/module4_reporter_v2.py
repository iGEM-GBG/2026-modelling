"""
Module 4 v2: Reporter Expression (Ste12 -> mRNA -> Reporter maturation)
===========================================================================
HILL-FUNCTION (cooperative) promoter response, replacing v1's linear
term. Scope note: ONLY the transcription step changes here -- Ste12
activation/deactivation (Fus3* -> Ste12*) stays exactly the same
mass-action 2-state switch as v1, and translation/maturation/
degradation are untouched. This mirrors Module 3 v2's "isolate one
kinetics-form change" approach.

    Fus3* + Ste12  --k_activate_Ste12-->  Ste12*      (Dig1/Dig2 released)
    Ste12*         --k_deactivate_Ste12--> Ste12        [unchanged, mass action]

    (basal) + Ste12* --Hill(n, K_half)-->  mRNA          [NEW in v2]
    mRNA           --k_degrade_mRNA-->  (degraded)

    mRNA           --k_translate-->     Reporter_immature (dark)
    Reporter_immature --k_mat-->        Reporter_mature   (fluorescent)
    Reporter_mature --k_degrade_mature--> (degraded)

Conservation law (Ste12 only -- unchanged from v1):
    Ste12_total = Ste12 + Ste12*

State vector: y = [Ste12_active, mRNA, Reporter_immature, Reporter_mature]

Output: [Reporter_mature](t) -- the measurable biosensor signal.

Chain: Module 3 v2 [Fus3*](t) --> text file --> interp1d --> input here
(reads module3_v2_output files, NOT v1's -- this is the v2 pipeline)

WHAT CHANGED FROM v1 -- THE KINETICS FORM AND ITS UNITS
-----------------------------------------------------------
v1:  d[mRNA]/dt = k_txn_basal + k_txn * Ste12*                  - decay
v2:  d[mRNA]/dt = k_txn_basal + k_txn_max * Ste12*^n/(K_half^n+Ste12*^n) - decay

Same units lesson as Module 3 v2: v1's k_txn has units s^-1 (multiplies
a concentration to give a rate). v2's Hill fraction is dimensionless,
so its prefactor needs units nM/s (a true Vmax) -- NOT the same number
reused from v1. Unit-correct rescaling, matching Module 3's approach:
    k_txn_max = k_txn(v1) * Ste12_total
This makes v2's ceiling (approached as Ste12* -> Ste12_total) equal
v1's ceiling (k_txn * Ste12_total) exactly -- an apples-to-apples
comparison of kinetics FORM, not accidentally also a comparison of
scale (see Module 3 v2's post-mortem on this exact mistake).

K_half is chosen differently from Module 3's Km on purpose: Module 3's
Km was set SMALL relative to the pool (0.1x) to produce ultrasensitive
switching. Here, K_half is set to the MIDDLE of Ste12*'s achievable
range (0.5 x Ste12_total) instead -- if K_half were small like Module
3's Km, the promoter would already be saturated at almost any nonzero
Ste12*, which (given Ste12* itself tends to saturate quickly) would
just recreate a flat response for a different reason.

n (Hill coefficient) = 2 -- a placeholder starting value (modest
cooperativity), NOT fit to real data yet. Per the earlier design
discussion: n is an upper bound set by the number of PREs, not
automatically equal to it (cooperativity is rarely perfect) -- treat
this as free to refit once real dose-response data exists.

MODELING STATUS (unchanged from v1, still applies)
----------------------------------------------------
* mRNA and both protein pools still have NO conservation law.
* Basal initial conditions are STILL valid in closed form here, even
  with a Hill promoter -- because Ste12* = 0 exactly at basal
  (Fus3* = 0 pre-stimulation collapses Ste12* to 0 regardless of
  kinetics form upstream), and the Hill numerator is 0^n = 0 for any
  n. Both v1's linear term and v2's Hill term vanish the same way at
  that specific point -- my earlier v1 comment claiming this would
  need numerical pre-equilibration in v2 was more cautious than
  necessary (see design discussion for the full correction).
* Reporter (mature protein) degradation still assumes a STABLE,
  non-degron reporter (7 h half-life, Mateus & Avery 2000) by default.
* All rate constants carry the same placeholder caveats as v1 -- see
  module4_reporter.py for full sourcing notes.

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

def load_module3_v2_output(L_nM):
    """Load [Fus3*](t) written by module3_mapk_v2.py for a given [L]."""
    path = os.path.join(INTERMEDIATE_DIR, f"module3_v2_output_L{L_nM:.1f}nM.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run modules 1, 2, then module3_mapk_v2.py "
            f"first to generate this module's input for [L] = {L_nM} nM."
        )
    data = np.loadtxt(path)
    return data[:, 0], data[:, 4]   # t, Fus3_active (column 4 -- see module3_v2 header)

# ──────────────────────────────────────────────────────────────
# PARAMETERS -- Ste12 switch and downstream steps FROZEN from v1
# ──────────────────────────────────────────────────────────────
Ste12_total = 105.5   # nM  -- PLACEHOLDER (SGD median abundance), frozen from v1

k_activate_Ste12   = 0.3   # nM^-1 s^-1  -- frozen from v1, UNCHANGED FORM
k_deactivate_Ste12 = 0.167   # s^-1        -- frozen from v1, UNCHANGED FORM

k_degrade_mRNA   = 5.78e-4   # s^-1  -- frozen from v1
k_translate      = 2e-2      # s^-1  -- frozen from v1
k_mat            = 7.70e-4   # s^-1  -- frozen from v1
k_degrade_mature = 2.75e-5   # s^-1  -- frozen from v1

k_transcribe_basal = 0.11  # nM/s  -- frozen from v1 (leaky transcription unchanged)

# --- NEW in v2: Hill-function promoter parameters -----------------------
# k_txn_max: unit-correct rescale of v1's k_txn, see docstring above.
k_transcribe_v1 = 5e-4   # s^-1  -- v1's linear constant, kept only for this rescale
k_txn_max = k_transcribe_v1 * Ste12_total   # nM/s

# K_half: centered in Ste12*'s achievable range (NOT small like Module 3's
# Km -- see docstring for why that distinction matters here).
K_half = 0.5 * Ste12_total   # nM  -- PLACEHOLDER, designed choice

# n: Hill coefficient, placeholder starting value -- free parameter to
# fit once real dose-response data exists (see docstring).
n_hill = 2   # dimensionless -- PLACEHOLDER

# ──────────────────────────────────────────────────────────────
# ODE DEFINITION
# ──────────────────────────────────────────────────────────────
def reporter_ode_v2(t, y, k_act12, k_deact12, k_txn_max_, K_half_, n_,
                     k_txn_basal, k_deg_mRNA, k_tln, k_mat_, k_deg_mat,
                     Ste12_tot, Fus3_interp):
    """
    Reporter expression ODEs -- v2 (Hill promoter, see MODELING STATUS).

    State vector:
        y[0] = Ste12_active
        y[1] = mRNA
        y[2] = Reporter_immature (dark)
        y[3] = Reporter_mature   (fluorescent -- the measured output)
    """
    Ste12a, mRNA, Rep_dark, Rep_mat = y

    Fus3a = max(float(Fus3_interp(t)), 0.0)   # numerical safety clamp
    Ste12a_c = max(Ste12a, 0.0)                # guard Hill term against overshoot

    dSte12a_dt = (k_act12 * Fus3a * (Ste12_tot - Ste12a) - k_deact12 * Ste12a)

    hill = Ste12a_c**n_ / (K_half_**n_ + Ste12a_c**n_)
    dmRNA_dt = (k_txn_basal + k_txn_max_ * hill - k_deg_mRNA * mRNA)

    dRepD_dt = (k_tln * mRNA - k_mat_ * Rep_dark)
    dRepM_dt = (k_mat_ * Rep_dark - k_deg_mat * Rep_mat)

    return [dSte12a_dt, dmRNA_dt, dRepD_dt, dRepM_dt]

# ──────────────────────────────────────────────────────────────
# INITIAL CONDITIONS -- basal steady state
# ──────────────────────────────────────────────────────────────
# STILL valid in closed form with a Hill promoter -- see MODELING STATUS
# for the corrected reasoning (Ste12*=0 at basal makes the Hill
# numerator 0 regardless of n, identical to v1's linear term at Ste12*=0).
mRNA_basal     = k_transcribe_basal / k_degrade_mRNA
Rep_dark_basal = k_translate * mRNA_basal / k_mat
Rep_mat_basal  = k_translate * mRNA_basal / k_degrade_mature
y0 = [0.0, mRNA_basal, Rep_dark_basal, Rep_mat_basal]

# ──────────────────────────────────────────────────────────────
# SIMULATION
# ──────────────────────────────────────────────────────────────
t_start  = 0
t_end    = 4 * 3600     # 4 hours, matching v1
n_points = 5000

L_values_nM = [1e5]
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(L_values_nM)))

results_m4 = {}

for L in L_values_nM:
    t_m3, Fus3_m3 = load_module3_v2_output(L)
    Fus3_interp = interp1d(t_m3, Fus3_m3, kind='cubic', fill_value='extrapolate')

    t_eval = np.linspace(t_start, t_end, n_points)
    sol = solve_ivp(
        fun    = reporter_ode_v2,
        t_span = (t_start, t_end),
        y0     = y0,
        t_eval = t_eval,
        args   = (k_activate_Ste12, k_deactivate_Ste12,
                  k_txn_max, K_half, n_hill,
                  k_transcribe_basal, k_degrade_mRNA,
                  k_translate, k_mat, k_degrade_mature,
                  Ste12_total, Fus3_interp),
        method = 'RK45',
        rtol   = 1e-8,
        atol   = 1e-10,
    )
    results_m4[L] = sol

    out_path = os.path.join(INTERMEDIATE_DIR, f"module4_v2_output_L{L:.1f}nM.txt")
    header = (
        "Module 4 v2 output -- Reporter expression (Hill promoter)\n"
        f"L = {L} nM ; Ste12_total = {Ste12_total} nM ; n = {n_hill} ; "
        f"K_half = {K_half} nM ; k_txn_max = {k_txn_max:.4e} nM/s\n"
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

ax1 = fig.add_subplot(gs[0, 0])
for (L, sol), c in zip(results_m4.items(), colors):
    ax1.plot(sol.t / 60, sol.y[3], color=c, lw=2, label=f"[L] = {L} nM")
ax1.set_xlabel("Time (min)")
ax1.set_ylabel("[Reporter$_{mature}$] (nM)")
ax1.set_title("A.  Fluorescent reporter over time (v2)\n(final biosensor readout)")
ax1.legend(fontsize=7, loc='upper left')
ax1.set_xlim(0, t_end / 60)
ax1.set_ylim(bottom=0)

ax2 = fig.add_subplot(gs[0, 1])
L_demo = 5.0
sol_demo = results_m4[L_demo]
t_min = sol_demo.t / 60
ax2.plot(t_min, sol_demo.y[0], lw=2, color='steelblue',  label='Ste12*')
ax2.plot(t_min, sol_demo.y[1], lw=2, color='darkorange', label='mRNA')
ax2.plot(t_min, sol_demo.y[2], lw=2, color='gray',       label='Reporter (dark)')
ax2.plot(t_min, sol_demo.y[3], lw=2, color='forestgreen', label='Reporter (mature)')
ax2.set_xlabel("Time (min)")
ax2.set_ylabel("Concentration (nM)")
ax2.set_title(f"B.  All four species (v2)\n([L] = {L_demo} nM = $K_D$)")
ax2.legend(fontsize=8)
ax2.set_xlim(0, t_end / 60)

ax3 = fig.add_subplot(gs[1, 0])
Rep_at_tend = [results_m4[L].y[3][-1] for L in L_values_nM]
ax3.semilogx(L_values_nM, Rep_at_tend, 'o-', color='forestgreen', lw=2, ms=8)
ax3.axvline(5.0, color='crimson', ls='--', lw=1.5, label='$K_D$ = 5 nM')
ax3.set_xlabel("[L] (nM, log scale)")
ax3.set_ylabel(f"[Reporter$_{{mature}}$] at t={t_end/3600:.0f}h (nM)")
ax3.set_title("C.  Reporter signal vs [L] (v2)\n(compare against v1's Panel C)")
ax3.legend(fontsize=9)
ax3.set_ylim(bottom=0)

# ── Panel D: Hill coefficient n sensitivity ──
ax4 = fig.add_subplot(gs[1, 1])
L_fixed = 5.0
t_m3, Fus3_m3 = load_module3_v2_output(L_fixed)
Fus3_interp_fixed = interp1d(t_m3, Fus3_m3, kind='cubic', fill_value='extrapolate')
t_eval = np.linspace(t_start, t_end, n_points)

n_variants = {'n=1 (no cooperativity)': 1, 'n=2 (this script)': 2,
              'n=4 (strong cooperativity)': 4}
variant_colors = ['#7f7f7f', '#2ca02c', '#d62728']

for (label, nv), c in zip(n_variants.items(), variant_colors):
    sol_v = solve_ivp(
        reporter_ode_v2, (t_start, t_end), y0, t_eval=t_eval,
        args=(k_activate_Ste12, k_deactivate_Ste12,
              k_txn_max, K_half, nv,
              k_transcribe_basal, k_degrade_mRNA,
              k_translate, k_mat, k_degrade_mature,
              Ste12_total, Fus3_interp_fixed),
        method='RK45', rtol=1e-8, atol=1e-10
    )
    ax4.plot(sol_v.t / 60, sol_v.y[3], lw=2, color=c, label=label)

ax4.set_xlabel("Time (min)")
ax4.set_ylabel("[Reporter$_{mature}$] (nM)")
ax4.set_title("D.  Hill coefficient n sensitivity\n"
              f"[L] = {L_fixed} nM = $K_D$")
ax4.legend(fontsize=7)
ax4.set_xlim(0, t_end / 60)

fig.suptitle(
    "Module 4 v2 -- Reporter Expression (Hill promoter, n=2, "
    "Ste12 switch unchanged)\n"
    f"K_half = 0.5 x Ste12_total [DESIGNED]  |  k_txn_max unit-rescaled "
    "from v1 (see script header)",
    fontsize=9, y=1.01
)

plt.savefig("../figures/module4_reporter_v2.png", dpi=150, bbox_inches='tight')
plt.close()
print("Figure saved.")

# ──────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ──────────────────────────────────────────────────────────────
print(f"\n── Signal at t = {t_end/3600:.0f}h (v2, Hill promoter) ─────────────────")
print(f"{'[L] (nM)':>10} {'[Ste12*]':>10} {'[mRNA]':>10} "
      f"{'[Rep_dark]':>12} {'[Rep_mature]':>13}")
print("-" * 60)
for L, sol in results_m4.items():
    s12 = sol.y[0][-1]
    m   = sol.y[1][-1]
    rd  = sol.y[2][-1]
    rm  = sol.y[3][-1]
    print(f"{L:>10.1f} {s12:>10.3f} {m:>10.3f} {rd:>12.3f} {rm:>13.3f}")

print("\n── Parameter status ──────────────────────────────────────────────")
print("  Ste12 switch, translation, maturation, degradation  <- FROZEN from v1")
print(f"  k_txn_max = {k_txn_max:.4e} nM/s  <- v1's k_txn unit-rescaled by Ste12_total")
print(f"  K_half    = {K_half} nM  <- DESIGNED (0.5 x Ste12_total), not measured")
print(f"  n         = {n_hill}  <- PLACEHOLDER starting value, fit once dose-response data exists")
