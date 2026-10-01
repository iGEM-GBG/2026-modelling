"""
Module 2: G-protein Activation Cycle
======================================
Models the heterotrimeric G-protein cycle driven by activated receptor [RL]:

    G_GDP  ---k_act*[RL]---> G_alpha,GTP + G_betagamma   (activation)
    G_alpha,GTP ---k_hyd---> G_alpha,GDP                  (hydrolysis)
    G_alpha,GDP + G_betagamma ---k_reassoc---> G_GDP      (reassociation)

Sst2 is DELETED in the biosensor chassis (Δsst2):
    k_RGS = 0  -->  hydrolysis rate = k_hyd only (no RGS acceleration)

Conservation law:
    G_total = [G_GDP] + [G_alpha,GTP] + [G_alpha,GDP] + [G_betagamma]
    [G_GDP] is recovered from conservation (not a state variable)

State vector: y = [G_alpha_GTP, G_alpha_GDP, G_betagamma]

Output to Module 3: [G_betagamma](t)  -->  drives Ste20 recruitment

Chain: Module 1 [RL](t) --> interp1d --> input to this module

Authors : iGEM team Gothenburg
Date    : 2026
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import os

# ──────────────────────────────────────────────────────────────
# PARAMETERS
# ──────────────────────────────────────────────────────────────
k_act     = 4e-4     # 1/(nM*s) -- Yi, Kitano & Simon (2003), native Ste2/Gpa1;
                      # primary tunable parameter for chimera coupling
k_hyd     = 0.004     # 1/s -- Yi et al. (2003)
k_reassoc = 1e-3    # 1/(nM*s) -- Kofahl & Klipp (2004) model value
G_total   = 200       # nM -- Gpa1 median abundance, Ho et al. 2018 unified
                      # dataset via SGD (5057 molecules/cell, 42 fL conversion)
k_RGS     = 0.0       # Sst2 deleted in biosensor chassis (Dsst2)

# ──────────────────────────────────────────────────────────────
# LOAD MODULE 1 OUTPUT (read from disk -- Module 1 must be run first)
# ──────────────────────────────────────────────────────────────
INTERMEDIATE_DIR = "../intermediate"
FIGURE_DIR = "../figures"
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

def load_module1_output(L_nM):
    """Load [RL](t) written by module1_binding.py for a given [L]."""
    path = os.path.join(INTERMEDIATE_DIR, f"module1_output_L{L_nM:.1f}nM.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run module1_binding.py first to "
            f"generate Module 1's output for [L] = {L_nM} nM."
        )
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]   # t, RL

# ──────────────────────────────────────────────────────────────
# ODE DEFINITION
# ──────────────────────────────────────────────────────────────
def gprotein_ode(t, y, k_act, k_hyd, k_reassoc, G_total, RL_interp):
    """
    G-protein cycle ODEs.

    State vector:
        y[0] = G_alpha_GTP   (active Galpha; also equals G_betagamma released)
        y[1] = G_alpha_GDP   (post-hydrolysis free Galpha)
        y[2] = G_betagamma   (free signal-carrying complex)

    G_GDP recovered from conservation:
        G_GDP = G_total - G_alpha_GTP - G_alpha_GDP - G_betagamma
    """
    G_aGTP  = y[0]
    G_aGDP  = y[1]
    Gbg     = y[2]

    # Recover G_GDP from conservation; clamp to zero to avoid
    # tiny negative values from numerical error
    G_GDP = max(G_total - G_aGTP - G_aGDP - Gbg, 0.0)

    # [RL] from Module 1 interpolant (continuous input)
    RL = float(RL_interp(t))
    RL = max(RL, 0.0)   # numerical safety clamp

    # Activation: G_GDP --k_act*[RL]--> G_alpha,GTP + G_betagamma
    v_act     = k_act * RL * G_GDP

    # Hydrolysis: G_alpha,GTP --k_hyd--> G_alpha,GDP  (k_RGS=0, Δsst2)
    v_hyd     = k_hyd * G_aGTP

    # Reassociation: G_alpha,GDP + G_betagamma --k_reassoc--> G_GDP
    v_reassoc = k_reassoc * G_aGDP * Gbg

    dG_aGTP_dt = v_act    - v_hyd
    dG_aGDP_dt = v_hyd    - v_reassoc
    dGbg_dt    = v_act    - v_reassoc

    return [dG_aGTP_dt, dG_aGDP_dt, dGbg_dt]

def dense_early_t_eval(t_start, t_end, dense_end=300, n_dense=3000, n_sparse=500):
    """Time grid concentrated in [t_start, dense_end] (where the real
    kinetics happen -- see design discussion: Module 2's own G-protein
    activation is the rate-limiting step, settling over ~1-2 min, not
    the much faster downstream MAPK relay). A uniform grid spread over
    the full multi-hour window is too coarse near t=0 to resolve this;
    this concatenates a dense early segment with a sparse late segment
    that just confirms the long-term plateau.
    """
    t_dense = np.linspace(t_start, dense_end, n_dense)
    t_sparse = np.linspace(dense_end, t_end, n_sparse)[1:]
    return np.concatenate([t_dense, t_sparse])

# ──────────────────────────────────────────────────────────────
# SIMULATION
# ──────────────────────────────────────────────────────────────
# LSODA (not RK45): the Kofahl & Klipp-sourced rate constants make
# this system numerically stiff (fast sub-processes alongside the
# hours-long simulation window); RK45 stalls, LSODA auto-switches
# to an implicit stiff method and solves it in milliseconds.
t_start  = 0
t_end    = 10800        # 3 hours

L_values_nM = [1e5]     # matching Module 1 (100 uM tyramine, wet-lab dose)
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(L_values_nM)))

results_m2 = {}

for L in L_values_nM:
    # --- Chain: get [RL](t) from Module 1's saved output ---
    t_m1, RL_m1 = load_module1_output(L)
    RL_interp = interp1d(t_m1, RL_m1, kind='cubic', fill_value='extrapolate')

    # --- Run Module 2 ---
    t_eval = dense_early_t_eval(t_start, t_end)
    sol = solve_ivp(
        fun    = gprotein_ode,
        t_span = (t_start, t_end),
        y0     = [0.0, 0.0, 0.0],   # all G-protein starts as G_GDP
        t_eval = t_eval,
        args   = (k_act, k_hyd, k_reassoc, G_total, RL_interp),
        method = 'RK45',
        rtol   = 1e-8,
        atol   = 1e-10,
    )
    results_m2[L] = sol

    # --- write [Gbg](t) (and the other two species) to disk for Module 3 ---
    out_path = os.path.join(INTERMEDIATE_DIR, f"module2_output_L{L:.1f}nM.txt")
    header = (
        "Module 2 output -- G-protein activation cycle (Dsst2 chassis)\n"
        f"L = {L} nM ; k_act = {k_act} 1/(nM*s) ; k_hyd = {k_hyd} 1/s ; "
        f"k_reassoc = {k_reassoc} 1/(nM*s) ; G_total = {G_total} nM\n"
        "columns: time_s   Ga_GTP_nM   Ga_GDP_nM   Gbg_nM"
    )
    np.savetxt(out_path,
               np.column_stack([sol.t, sol.y[0], sol.y[1], sol.y[2]]),
               header=header, fmt="%.6e")

L_demo = L_values_nM[0]

# ──────────────────────────────────────────────────────────────
# PLOTTING -- each panel saved as its own PDF
# ──────────────────────────────────────────────────────────────

# ── A: G_betagamma(t) -- signal output ───────────────────────
figA, axA = plt.subplots(figsize=(6.5, 5))
for (L, sol), c in zip(results_m2.items(), colors):
    axA.plot(sol.t / 60, sol.y[2], color=c, lw=2, label=f"[L] = {L:.0f} nM")
axA.set_xlabel("Time (min)")
axA.set_ylabel("[G\u03b2\u03b3] (nM)")
axA.set_title("Free G\u03b2\u03b3 over time\n(signal input to Module 3)")
axA.legend(fontsize=9, loc='upper left')
axA.set_xlim(0, 10)   # zoomed: real kinetics settle within ~2 min (see design discussion)
axA.set_ylim(bottom=0)
figA.tight_layout()
figA.savefig(os.path.join(FIGURE_DIR, "module2_A_Gbg_timecourse.pdf"))
plt.close(figA)

# ── B: All three species at the simulated dose ───────────────
figB, axB = plt.subplots(figsize=(6.5, 5))
sol_demo = results_m2[L_demo]
t_min = sol_demo.t / 60
axB.plot(t_min, sol_demo.y[0], lw=2, color='steelblue',  label='$G_{\\alpha,GTP}$')
axB.plot(t_min, sol_demo.y[1], lw=2, color='darkorange', label='$G_{\\alpha,GDP}$')
axB.plot(t_min, sol_demo.y[2], lw=2, color='forestgreen', label='$G_{\\beta\\gamma}$')
G_GDP_demo = G_total - sol_demo.y[0] - sol_demo.y[1] - sol_demo.y[2]
axB.plot(t_min, G_GDP_demo, lw=2, color='crimson', ls='--', label='$G_{GDP}$ (conserved)')
axB.set_xlabel("Time (min)")
axB.set_ylabel("Concentration (nM)")
axB.set_title(f"All G-protein species\n([L] = {L_demo:.0f} nM, wet-lab dose)")
axB.legend(fontsize=9)
axB.set_xlim(0, 10)   # zoomed: real kinetics settle within ~2 min (see design discussion)
axB.set_ylim(bottom=0)
figB.tight_layout()
figB.savefig(os.path.join(FIGURE_DIR, "module2_B_all_species.pdf"))
plt.close(figB)

# ── C: k_act sensitivity -- effect of chimera coupling strength ──
figC, axC = plt.subplots(figsize=(6.5, 5))
t_m1, RL_m1 = load_module1_output(L_demo)
RL_interp_fixed = interp1d(t_m1, RL_m1, kind='cubic', fill_value='extrapolate')
t_eval = dense_early_t_eval(t_start, t_end)

k_act_variants = {
    '0.1x (weak chimera)':  k_act * 0.1,
    '1x (baseline)':        k_act,
    '5x (strong chimera)':  k_act * 5,
    '20x (strong chimera)': k_act * 20,
}
variant_colors = ['#d62728', '#7f7f7f', '#2ca02c', '#1f77b4']

for (label, ka), c in zip(k_act_variants.items(), variant_colors):
    sol_v = solve_ivp(
        gprotein_ode, (t_start, t_end), [0.0, 0.0, 0.0],
        t_eval=t_eval,
        args=(ka, k_hyd, k_reassoc, G_total, RL_interp_fixed),
        method='LSODA', rtol=1e-8, atol=1e-10
    )
    axC.plot(sol_v.t / 60, sol_v.y[2], lw=2, color=c, label=label)

axC.set_xlabel("Time (min)")
axC.set_ylabel("[G\u03b2\u03b3] (nM)")
axC.set_title(f"Chimera coupling sensitivity (k$_{{act}}$ variants)\n"
              f"[L] = {L_demo:.0f} nM")
axC.legend(fontsize=9)
axC.set_xlim(0, 10)   # zoomed: real kinetics settle within ~2 min (see design discussion)
axC.set_ylim(bottom=0)
figC.tight_layout()
figC.savefig(os.path.join(FIGURE_DIR, "module2_C_kact_sensitivity.pdf"))
plt.close(figC)

print("Figures saved: module2_{A_Gbg_timecourse,B_all_species,"
      "C_kact_sensitivity}.pdf")

# ──────────────────────────────────────────────────────────────
# STEADY-STATE SUMMARY
# ──────────────────────────────────────────────────────────────
print("\n-- Steady-state summary ----------------------------------------")
print(f"{'[L] (nM)':>10} {'[Gbg]_ss (nM)':>15} {'% of G_total':>14} "
      f"{'[GaGTP]_ss':>12} {'[GaGDP]_ss':>12}")
print("-" * 67)
for L, sol in results_m2.items():
    Gbg   = sol.y[2][-1]
    GaGTP = sol.y[0][-1]
    GaGDP = sol.y[1][-1]
    pct   = 100 * Gbg / G_total
    print(f"{L:>10.1f} {Gbg:>15.3f} {pct:>13.1f}% {GaGTP:>12.3f} {GaGDP:>12.3f}")