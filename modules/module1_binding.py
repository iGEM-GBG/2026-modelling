"""
Module 1: Ligand-Receptor Binding Kinetics
===========================================
Models the reversible binding reaction:

    R + L <--k_on--> RL
           <--k_off--

System: TAAR1 (trace amine-associated receptor 1) + Tyramine
        expressed in S. cerevisiae biosensor chassis

ODEs
----
d[RL]/dt = k_on * [R] * [L] - k_off * [RL]
[R]      = R_total - [RL]          (conservation law; no free R ODE needed)
[L]      = constant                (ligand not depleted; large excess assumed)

Output fed to Module 2: [RL](t)  -->  k_act * [RL] drives G-protein activation

Authors : iGEM team Gothenburg 2026
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ──────────────────────────────────────────────────────────────
# INTERMEDIATE OUTPUT (read by Module 2)
# ──────────────────────────────────────────────────────────────
# Module 2 (and later modules) do NOT import this file or re-run its
# ODE. Instead, this script writes [RL](t) to a plain-text file per
# ligand condition, which the next module reads and interpolates.
# Re-run this script after changing any parameter so Module 2's
# inputs don't go stale.
INTERMEDIATE_DIR = "../intermediate"
FIGURE_DIR = "../figures"
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# PARAMETERS
# ──────────────────────────────────────────────────────────────
k_on  = 1.22e-3      # 1/(nM*s) -- Tran, Chang & Snyder (1978)
K_D   = 20            # nM -- Borowsky et al. (2001)
k_off = K_D * k_on   # 1/s -- derived (K_D * k_on)

# Explicit upper bound on functionally available receptor, not a
# central estimate: total expressed receptor (~1,000 molecules/cell,
# 42 fL conversion); fluorescence microscopy showed predominantly
# ER/vacuolar retention, so true membrane-available receptor is very
# likely lower by an unquantified factor.
R_total = 50.0       # nM

# ──────────────────────────────────────────────────────────────
# SIMULATION SETTINGS
# ──────────────────────────────────────────────────────────────
t_start = 0
t_end   = 10800       # 3 hours
n_points = 1000

L_values_nM = [1e5]   # 100 uM tyramine -- the actual wet-lab dose

# ──────────────────────────────────────────────────────────────
# ODE DEFINITION
# ──────────────────────────────────────────────────────────────
def binding_ode(t, y, k_on, k_off, R_total, L):
    """
    ODE for receptor-ligand binding.

    State vector y = [RL]
    L is treated as a constant (large ligand excess; not depleted).
    R = R_total - RL  (conservation)
    """
    RL = y[0]
    R  = R_total - RL
    dRL_dt = k_on * R * L - k_off * RL
    return [dRL_dt]

# ──────────────────────────────────────────────────────────────
# ANALYTICAL STEADY STATE (for validation)
# ──────────────────────────────────────────────────────────────
def RL_steady_state(L, R_total, K_D):
    """Hill-Langmuir isotherm (n=1): [RL]_ss = R_total*L / (K_D + L)"""
    return R_total * L / (K_D + L)

# ──────────────────────────────────────────────────────────────
# RUN SIMULATIONS
# ──────────────────────────────────────────────────────────────
t_eval = np.linspace(t_start, t_end, n_points)
results = {}

for L in L_values_nM:
    sol = solve_ivp(
        fun      = binding_ode,
        t_span   = (t_start, t_end),
        y0       = [0.0],           # start with zero bound receptor
        t_eval   = t_eval,
        args     = (k_on, k_off, R_total, L),
        method   = 'RK45',
        rtol     = 1e-8,
        atol     = 1e-10,
    )
    results[L] = sol

    # --- write [RL](t) to disk for Module 2 to read ---
    out_path = os.path.join(INTERMEDIATE_DIR, f"module1_output_L{L:.1f}nM.txt")
    header = (
        "Module 1 output -- HRH4/Histamine binding kinetics\n"
        f"L = {L} nM ; K_D = {K_D} nM ; k_on = {k_on} 1/(nM*s) ; "
        f"k_off = {k_off} 1/s ; R_total = {R_total} nM\n"
        "columns: time_s   RL_nM"
    )
    np.savetxt(out_path, np.column_stack([sol.t, sol.y[0]]),
               header=header, fmt="%.6e")

# Dose-response: steady-state [RL] across a dense L range (analytical;
# independent of how many doses were actually simulated above)
L_range     = np.logspace(-1, 6, 400)   # 0.1 nM to 1,000,000 nM
RL_ss_curve = RL_steady_state(L_range, R_total, K_D)
fractional  = RL_ss_curve / R_total

# ──────────────────────────────────────────────────────────────
# PLOTTING -- each panel saved as its own PDF
# ──────────────────────────────────────────────────────────────
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(L_values_nM)))

# ── A: [RL](t) time courses ──────────────────────────────────
figA, axA = plt.subplots(figsize=(6.5, 5))
for (L, sol), c in zip(results.items(), colors):
    RL_t  = sol.y[0]
    RL_ss = RL_steady_state(L, R_total, K_D)
    axA.plot(sol.t / 60, RL_t, color=c, lw=2, label=f"[L] = {L:.0f} nM")
    axA.axhline(RL_ss, color=c, lw=1, ls='--', alpha=0.5)
axA.set_xlabel("Time (min)")
axA.set_ylabel("[RL] (nM)")
axA.set_title("Binding kinetics -- [RL](t)\n(dashed = analytical steady state)")
axA.legend(fontsize=9, loc='upper left')
axA.set_xlim(0, t_end / 60)
axA.set_ylim(bottom=0)
figA.tight_layout()
figA.savefig(os.path.join(FIGURE_DIR, "module1_A_binding_timecourse.pdf"))
plt.close(figA)

# ── B: Fractional occupancy (dose-response) ──────────────────
figB, axB = plt.subplots(figsize=(6.5, 5))
axB.semilogx(L_range, fractional, 'k', lw=2.5)
axB.axvline(K_D, color='crimson', ls='--', lw=1.5, label=f'$K_D$ = {K_D} nM')
axB.axhline(0.5, color='crimson', ls=':', lw=1.0)
for L, c in zip(L_values_nM, colors):
    occ = RL_steady_state(L, R_total, K_D) / R_total
    axB.scatter(L, occ, color=c, zorder=5, s=70, label=f"simulated: [L]={L:.0f} nM")
axB.set_xlabel("[L] (nM, log scale)")
axB.set_ylabel("Fractional occupancy [RL] / R$_{total}$")
axB.set_title("Dose-response (Hill-Langmuir, n = 1)")
axB.legend(fontsize=9)
axB.set_ylim(0, 1.05)
figB.tight_layout()
figB.savefig(os.path.join(FIGURE_DIR, "module1_B_dose_response.pdf"))
plt.close(figB)

# ── C: Linear regime detail (L << K_D) ────────────────────────
figC, axC = plt.subplots(figsize=(6.5, 5))
L_linear = np.linspace(0, K_D * 2, 200)
RL_linear = RL_steady_state(L_linear, R_total, K_D)
linear_approx = R_total / K_D * L_linear
axC.plot(L_linear, RL_linear,     'k',         lw=2,   label='Full model')
axC.plot(L_linear, linear_approx, 'steelblue', lw=1.5, ls='--',
         label='Linear approx. ($[L] \\ll K_D$)')
axC.axvline(K_D, color='crimson', ls='--', lw=1, label=f'$K_D$ = {K_D} nM')
axC.set_xlabel("[L] (nM)")
axC.set_ylabel("[RL] (nM)")
axC.set_title("Linear detection regime\n(useful range for quantitative sensing)")
axC.legend(fontsize=9)
figC.tight_layout()
figC.savefig(os.path.join(FIGURE_DIR, "module1_C_linear_regime.pdf"))
plt.close(figC)

# ── D: Equilibration time constant vs L ───────────────────────
figD, axD = plt.subplots(figsize=(6.5, 5))
L_tau = np.logspace(-1, 6, 300)
tau   = 1.0 / (k_on * L_tau + k_off)
axD.loglog(L_tau, tau / 60, 'darkorange', lw=2)
axD.axvline(K_D, color='crimson', ls='--', lw=1.5, label=f'$K_D$ = {K_D} nM')
for L, c in zip(L_values_nM, colors):
    axD.scatter(L, 1.0/(k_on*L+k_off)/60, color=c, zorder=5, s=70,
                label=f"simulated: [L]={L:.0f} nM")
axD.set_xlabel("[L] (nM, log scale)")
axD.set_ylabel("Time constant tau (min, log scale)")
axD.set_title("Equilibration time constant tau\n(= 1 / (k$_{on}$[L] + k$_{off}$))")
axD.legend(fontsize=9)
figD.tight_layout()
figD.savefig(os.path.join(FIGURE_DIR, "module1_D_equilibration_time.pdf"))
plt.close(figD)

print("Figures saved: module1_{A_binding_timecourse,B_dose_response,"
      "C_linear_regime,D_equilibration_time}.pdf")

# ──────────────────────────────────────────────────────────────
# STEADY-STATE SUMMARY
# ──────────────────────────────────────────────────────────────
print("\n-- Steady-state summary --------------------------------------")
print(f"{'[L] (nM)':>12} {'[RL]_ss (nM)':>14} {'Occupancy (%)':>15} {'tau (s)':>8}")
print("-" * 55)
for L in L_values_nM:
    RL_ss = RL_steady_state(L, R_total, K_D)
    occ   = 100 * RL_ss / R_total
    tau_v = 1.0 / (k_on * L + k_off)
    print(f"{L:>12.1f} {RL_ss:>14.2f} {occ:>14.1f}% {tau_v:>8.1f}")