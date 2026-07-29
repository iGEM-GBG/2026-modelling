"""
Module 1: Ligand-Receptor Binding Kinetics
===========================================
Models the reversible binding reaction:

    R + L <--k_on--> RL
           <--k_off--

System: HRH4 (human histamine H4 receptor) + Histamine
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

# ──────────────────────────────────────────────────────────────
# PARAMETERS
# ──────────────────────────────────────────────────────────────
#
# k_on  [1/(nM·s)]
#   Source: Motulsky & Bhave (2004) estimated k_on for GPCRs
#   typically in the range 1e-4 to 1e-2 1/(nM·s).
#   We use 1e-3 as a mid-range literature anchor.
#   *** PLACEHOLDER
k_on = 1e-3          # 1/(nM·s)

# k_off [1/s]
#   Derived from K_D = k_off / k_on.
#   K_D for HRH4/histamine: Lim et al. (2005) J Pharmacol Exp Ther
#   313(2):771-777 reported K_D ~ 5 nM using radioligand binding.
#   Cross-checked against IUPHAR Guide to Pharmacology HRH4 entry.
#   k_off = K_D * k_on = 5 nM * 1e-3 1/(nM·s) = 5e-3 1/s
#   *** k_on is a PLACEHOLDER; k_off inherits that uncertainty ***
K_D   = 5.0          # nM  — PLACEHOLDER, possible source Lim et al. 2005
k_off = K_D * k_on   # 5e-3  1/s  — derived

# R_total [nM]
#   Receptor copy number in engineered yeast ~ 1,000–10,000 molecules/cell.
#   Converted to nM assuming yeast cell volume ~ 42 fL (Ghaemmaghami 2003).
#   1,000 molecules / (42e-15 L * 6.022e23) ~ 40 nM; we use 50 nM.
#   *** PLACEHOLDER — replace with 26A1 epitope-tagging quantification ***
R_total = 50.0       # nM

# ──────────────────────────────────────────────────────────────
# SIMULATION SETTINGS
# ──────────────────────────────────────────────────────────────
t_start = 0
t_end   = 3600       # 1 hour in seconds, typical biosensor assay window
n_points = 1000

# Ligand concentrations to simulate [nM]
# Spans below K_D, around K_D, and above K_D to capture all regimes
L_values_nM = [0.5, 2.0, 5.0, 20.0, 100.0, 500.0]

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

# Dose-response: steady-state [RL] across a dense L range
L_range    = np.logspace(-1, 4, 400)   # 0.1 nM to 10,000 nM
RL_ss_curve = RL_steady_state(L_range, R_total, K_D)
fractional  = RL_ss_curve / R_total    # fractional occupancy

# ──────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(L_values_nM)))

# ── Panel A: [RL](t) time courses ────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
for (L, sol), c in zip(results.items(), colors):
    RL_t  = sol.y[0]
    RL_ss = RL_steady_state(L, R_total, K_D)
    label = f"[L] = {L} nM"
    ax1.plot(sol.t / 60, RL_t, color=c, lw=2, label=label)
    ax1.axhline(RL_ss, color=c, lw=1, ls='--', alpha=0.5)

ax1.set_xlabel("Time (min)")
ax1.set_ylabel("[RL] (nM)")
ax1.set_title("A.  Binding kinetics — [RL](t)\n(dashed = analytical steady state)")
ax1.legend(fontsize=7, loc='upper left')
ax1.set_xlim(0, t_end / 60)
ax1.set_ylim(bottom=0)

# ── Panel B: Fractional occupancy (dose-response) ─────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.semilogx(L_range, fractional, 'k', lw=2.5)
ax2.axvline(K_D, color='crimson', ls='--', lw=1.5, label=f'$K_D$ = {K_D} nM')
ax2.axhline(0.5,  color='crimson', ls=':',  lw=1.0)

# Mark simulated L values
for L, c in zip(L_values_nM, colors):
    occ = RL_steady_state(L, R_total, K_D) / R_total
    ax2.scatter(L, occ, color=c, zorder=5, s=60)

ax2.set_xlabel("[L] (nM, log scale)")
ax2.set_ylabel("Fractional occupancy [RL] / R$_{total}$")
ax2.set_title("B.  Dose-response (Hill-Langmuir, n = 1)")
ax2.legend(fontsize=9)
ax2.set_ylim(0, 1.05)

# ── Panel C: Linear regime detail (L << K_D) ──────────────────
ax3 = fig.add_subplot(gs[1, 0])
L_linear = np.linspace(0, K_D * 2, 200)
RL_linear = RL_steady_state(L_linear, R_total, K_D)
linear_approx = R_total / K_D * L_linear   # [RL] ≈ R_total/K_D * L

ax3.plot(L_linear, RL_linear,     'k',        lw=2,   label='Full model')
ax3.plot(L_linear, linear_approx, 'steelblue', lw=1.5, ls='--',
         label='Linear approx. ($[L] \\ll K_D$)')
ax3.axvline(K_D, color='crimson', ls='--', lw=1, label=f'$K_D$ = {K_D} nM')
ax3.set_xlabel("[L] (nM)")
ax3.set_ylabel("[RL] (nM)")
ax3.set_title("C.  Linear detection regime\n(useful range for quantitative sensing)")
ax3.legend(fontsize=8)

# ── Panel D: Time-to-equilibrium vs L ─────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
# Theoretical equilibration time constant: tau = 1 / (k_on*L + k_off)
L_tau = np.logspace(-1, 4, 300)
tau   = 1.0 / (k_on * L_tau + k_off)

ax4.loglog(L_tau, tau / 60, 'darkorange', lw=2)
ax4.axvline(K_D, color='crimson', ls='--', lw=1.5, label=f'$K_D$ = {K_D} nM')
ax4.set_xlabel("[L] (nM, log scale)")
ax4.set_ylabel("Time constant τ (min, log scale)")
ax4.set_title("D.  Equilibration time constant τ\n(= 1 / (k$_{{on}}$[L] + k$_{{off}}$))")
ax4.legend(fontsize=9)

# ── Main title ────────────────────────────────────────────────
fig.suptitle(
    "Module 1 — HRH4/Histamine Binding Kinetics\n"
    f"K$_D$ = {K_D} nM (Lim et al. 2005)  |  "
    f"k$_{{on}}$ = {k_on:.0e} nM⁻¹s⁻¹ [PLACEHOLDER]  |  "
    f"R$_{{total}}$ = {R_total} nM [PLACEHOLDER]",
    fontsize=11, y=1.01
)

plt.savefig("../figures/module1_binding.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Figure saved.")

# ──────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ──────────────────────────────────────────────────────────────
print("\n── Steady-state summary ──────────────────────────────────")
print(f"{'[L] (nM)':>12} {'[RL]_ss (nM)':>14} {'Occupancy (%)':>15} {'τ (s)':>8}")
print("-" * 55)
for L in L_values_nM:
    RL_ss = RL_steady_state(L, R_total, K_D)
    occ   = 100 * RL_ss / R_total
    tau   = 1.0 / (k_on * L + k_off)
    print(f"{L:>12.1f} {RL_ss:>14.2f} {occ:>14.1f}% {tau:>8.1f}")

print("\n── Parameter status ──────────────────────────────────────")
print(f"  K_D      = {K_D} nM      <- PLACEHOLDER Literature (Lim et al. 2005, IUPHAR)")
print(f"  k_on     = {k_on} nM⁻¹s⁻¹  <- PLACEHOLDER (typical GPCR range)")
print(f"  k_off    = {k_off} s⁻¹    <- Derived (k_off = K_D * k_on)")
print(f"  R_total  = {R_total} nM       <- PLACEHOLDER (replace with 26A1 data)")
