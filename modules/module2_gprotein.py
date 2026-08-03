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
#sys.path.insert(0, '/home/claude')

# ──────────────────────────────────────────────────────────────
# PARAMETERS
# ──────────────────────────────────────────────────────────────
#
# k_act  [1/(nM·s)]
#   Rate of receptor-catalysed GDP->GTP exchange on Gpa1.
#   Source: Yi, Kitano & Simon (2003) PNAS 100(19):10764-10769
#   measured k_act ~ 1e-3 1/(molecule/cell * s) for native Ste2/Gpa1.
#   Converted to nM units using yeast cell volume 42 fL
#   (Jorgensen et al. 2002, Science 297:395): 1 molecule/cell ~ 40 nM
#   k_act = 4e-4 1/(nM·s)  [literature, yeast native pathway]
#   *** PLACEHOLDER for chimera variants, fit to 26A3/26A6 data ***
#   Different Ga/Gpa1 chimeras will shift this value; it is the primary
#   tunable parameter for chimera optimisation.
k_act = 4e-4          # 1/(nM·s)

# k_hyd  [1/s]
#   Intrinsic GTPase rate of Gpa1 (Sst2-independent).
#   Source: Yi, Kitano & Simon (2003) PNAS 100(19):10764-10769
#   measured k_hyd = 0.004 1/s for yeast Gpa1.
#   Note: Sst2 is DELETED (k_RGS = 0), so this is the only hydrolysis term.
#   Literature value, reliable for Δsst2 chassis.
k_hyd = 0.004         # 1/s   -- Yi et al. 2003 (check literature)

# k_reassoc  [1/(nM·s)]
#   Rate of G_alpha,GDP + G_betagamma -> G_GDP heterotrimer reformation.
#   Source: Yi, Kitano & Simon (2003) PNAS 100(19):10764-10769
#   k_reassoc = 1e-3 1/(nM·s)
k_reassoc = 1e-3      # 1/(nM·s)  -- Yi et al. 2003 (check literature)

# G_total  [nM]
#   Total Gpa1/Gbeta/Ggamma pool.
#   Native Gpa1 (GPA1): ~1800 molecules/cell from Ghaemmaghami et al.
#   (2003) Nature 425:737. Converted: 1800 / (6.022e23 * 42e-15) ~ 71 nM
#   *** PLACEHOLDER — use your strain's actual expression level ***
G_total = 70.0        # nM   -- Ghaemmaghami et al. 2003 (placeholder)

# k_RGS = 0  (Sst2 deleted in biosensor chassis)
# Reference: Ehrenworth et al. 2017 Biochemistry; thesis Figure 6C
k_RGS = 0.0

# ──────────────────────────────────────────────────────────────
# LOAD MODULE 1 OUTPUT (read from disk -- Module 1 must be run first)
# ──────────────────────────────────────────────────────────────
# Module 2 no longer re-implements Module 1's ODE. It reads the
# [RL](t) text file that module1_binding.py wrote for each ligand
# condition. This is the single source of truth for Module 1's
# parameters/behaviour -- if you change something in Module 1,
# re-run it before re-running Module 2.
INTERMEDIATE_DIR = "../intermediate"
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

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

# ──────────────────────────────────────────────────────────────
# SIMULATION
# ──────────────────────────────────────────────────────────────
t_start  = 0
t_end    = 3600        # 1 hour
n_points = 2000

# Ligand concentrations matching Module 1
L_values_nM = [0.5, 2.0, 5.0, 20.0, 100.0, 500.0]
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(L_values_nM)))

results_m2 = {}

for L in L_values_nM:
    # --- Chain: get [RL](t) from Module 1's saved output ---
    t_m1, RL_m1 = load_module1_output(L)
    RL_interp = interp1d(t_m1, RL_m1, kind='cubic', fill_value='extrapolate')

    # --- Run Module 2 ---
    t_eval = np.linspace(t_start, t_end, n_points)
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

# Amplification factor across L values
# Amplif = k_act * [G_GDP] / k_off  (per-RL-molecule rate / dissociation rate)
k_off   = 5e-3    # from Module 1
L_range = np.array(L_values_nM)

# ──────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)

# ── Panel A: G_betagamma(t) — signal output ──────────────────
ax1 = fig.add_subplot(gs[0, 0])
for (L, sol), c in zip(results_m2.items(), colors):
    Gbg = sol.y[2]
    ax1.plot(sol.t / 60, Gbg, color=c, lw=2, label=f"[L] = {L} nM")
ax1.set_xlabel("Time (min)")
ax1.set_ylabel("[Gβγ] (nM)")
ax1.set_title("A.  Free Gβγ over time\n(signal input to Module 3)")
ax1.legend(fontsize=7, loc='upper left')
ax1.set_xlim(0, t_end / 60)
ax1.set_ylim(bottom=0)

# ── Panel B: All three species for [L] = 5 nM = K_D ─────────
ax2 = fig.add_subplot(gs[0, 1])
L_demo = 5.0
sol_demo = results_m2[L_demo]
t_min = sol_demo.t / 60
ax2.plot(t_min, sol_demo.y[0], lw=2, color='steelblue',  label='$G_{\\alpha,GTP}$')
ax2.plot(t_min, sol_demo.y[1], lw=2, color='darkorange',  label='$G_{\\alpha,GDP}$')
ax2.plot(t_min, sol_demo.y[2], lw=2, color='forestgreen', label='$G_{\\beta\\gamma}$')
G_GDP_demo = G_total - sol_demo.y[0] - sol_demo.y[1] - sol_demo.y[2]
ax2.plot(t_min, G_GDP_demo,    lw=2, color='crimson', ls='--', label='$G_{GDP}$ (conserved)')
ax2.set_xlabel("Time (min)")
ax2.set_ylabel("Concentration (nM)")
ax2.set_title(f"B.  All G-protein species\n([L] = {L_demo} nM = $K_D$)")
ax2.legend(fontsize=8)
ax2.set_xlim(0, t_end / 60)
ax2.set_ylim(bottom=0)

# ── Panel C: Steady-state Gbg vs L (dose-response) ───────────
ax3 = fig.add_subplot(gs[1, 0])
Gbg_ss = [results_m2[L].y[2][-1] for L in L_values_nM]
ax3.semilogx(L_values_nM, Gbg_ss, 'o-', color='forestgreen', lw=2, ms=8)
ax3.axvline(5.0, color='crimson', ls='--', lw=1.5, label='$K_D$ = 5 nM')
ax3.set_xlabel("[L] (nM, log scale)")
ax3.set_ylabel("[Gβγ]$_{ss}$ (nM)")
ax3.set_title("C.  Steady-state [Gβγ] vs [L]\n(Module 2 dose-response)")
ax3.legend(fontsize=9)
ax3.set_ylim(bottom=0)

# ── Panel D: k_act sensitivity — effect of chimera variants ──
ax4 = fig.add_subplot(gs[1, 1])
L_fixed = 5.0   # at K_D
t_m1, RL_m1 = load_module1_output(L_fixed)
RL_interp_fixed = interp1d(t_m1, RL_m1, kind='cubic', fill_value='extrapolate')
t_eval = np.linspace(t_start, t_end, n_points)

k_act_variants = {
    '0.1× (weak chimera)':  k_act * 0.1,
    '1× (baseline)':        k_act,
    '5× (strong chimera)':  k_act * 5,
    '20× (strong chimera)': k_act * 20,
}
variant_colors = ['#d62728', '#7f7f7f', '#2ca02c', '#1f77b4']

for (label, ka), c in zip(k_act_variants.items(), variant_colors):
    sol_v = solve_ivp(
        gprotein_ode, (t_start, t_end), [0.0, 0.0, 0.0],
        t_eval=t_eval,
        args=(ka, k_hyd, k_reassoc, G_total, RL_interp_fixed),
        method='RK45', rtol=1e-8, atol=1e-10
    )
    ax4.plot(sol_v.t / 60, sol_v.y[2], lw=2, color=c, label=label)

ax4.set_xlabel("Time (min)")
ax4.set_ylabel("[Gβγ] (nM)")
ax4.set_title(f"D.  Chimera sensitivity (k$_{{act}}$ variants)\n[L] = {L_fixed} nM = $K_D$")
ax4.legend(fontsize=8)
ax4.set_xlim(0, t_end / 60)
ax4.set_ylim(bottom=0)

fig.suptitle(
    "Module 2 — G-protein Activation Cycle (Δsst2 chassis)\n"
    f"k$_{{act}}$ = {k_act:.0e} nM⁻¹s⁻¹ [PLACEHOLDER: fit to 26A3/A6]  |  "
    f"k$_{{hyd}}$ = {k_hyd} s⁻¹ (Yi et al. 2003)  |  "
    f"G$_{{total}}$ = {G_total} nM [PLACEHOLDER]",
    fontsize=10, y=1.01
)

plt.savefig("../figures/module2_gprotein.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Figure saved.")

# ──────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ──────────────────────────────────────────────────────────────
print("\n── Steady-state summary ─────────────────────────────────────────")
print(f"{'[L] (nM)':>10} {'[Gbg]_ss (nM)':>15} {'% of G_total':>14} "
      f"{'[GaGTP]_ss':>12} {'[GaGDP]_ss':>12}")
print("-" * 67)
for L, sol in results_m2.items():
    Gbg   = sol.y[2][-1]
    GaGTP = sol.y[0][-1]
    GaGDP = sol.y[1][-1]
    pct   = 100 * Gbg / G_total
    print(f"{L:>10.1f} {Gbg:>15.3f} {pct:>13.1f}% {GaGTP:>12.3f} {GaGDP:>12.3f}")

print("\n── Parameter status ──────────────────────────────────────────────")
print(f"  k_act     = {k_act:.0e} nM⁻¹s⁻¹  ← PLACEHOLDER (Yi et al. 2003 native; "
      "fit to 26A3/A6 for chimeras)")
print(f"  k_hyd     = {k_hyd} s⁻¹      ← PLACEHOLDER, Yi et al. 2003 PNAS")
print(f"  k_reassoc = {k_reassoc:.0e} nM⁻¹s⁻¹  ← PLACEHOLDER, Yi et al. 2003 PNAS")
print(f"  G_total   = {G_total} nM        ← PLACEHOLDER (Ghaemmaghami 2003; "
      "replace with 26A1 quantification)")
print(f"  k_RGS     = {k_RGS}            ← Zero (Δsst2 chassis; Ehrenworth 2017)")