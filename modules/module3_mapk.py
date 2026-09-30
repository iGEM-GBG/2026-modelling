"""
Module 3: MAPK Cascade (Ste20 -> Ste11 -> Ste7 -> Fus3)
==========================================================
Models the four-tier phosphorylation relay downstream of the G-protein
cycle, using a v1 MASS-ACTION approximation (no saturation kinetics
yet -- see "MODELING STATUS" below):

    G_betagamma + Ste20   --k_activate_Ste20-->  Ste20*
    Ste20*      + Ste11   --k_activate_Ste11-->  Ste11*
    Ste11*      + Ste7    --k_activate_Ste7-->   Ste7*
    Ste7*       + Fus3    --k_activate_Fus3-->   Fus3*

    Ste20* --k_deactivate_Ste20--> Ste20   (etc. for each tier)

Each tier is a 2-state lumped switch (active / inactive), NOT a
3-state distributive dual-phosphorylation model, and NOT a
Goldbeter-Koshland saturating (Michaelis-Menten) switch.

Conservation laws (one per tier):
    Ste20_total = Ste20 + Ste20*
    Ste11_total = Ste11 + Ste11*
    Ste7_total  = Ste7  + Ste7*
    Fus3_total  = Fus3  + Fus3*

State vector: y = [Ste20_active, Ste11_active, Ste7_active, Fus3_active]

Output to Module 4: [Fus3*](t)  -->  drives Ste12 activation

Chain: Module 2 [Gbetagamma](t) --> text file --> interp1d --> input here

MODELING STATUS (read before trusting the dynamics)
----------------------------------------------------
* v1 = mass action. This deliberately OMITS the zero-order
  ultrasensitivity that motivated using Goldbeter-Koshland kinetics
  in the first place (see design discussion). This cascade will
  behave as a graded amplifier, not a switch, until saturating
  (Michaelis-Menten) terms are added in v2.
* Ste5 scaffold is NOT modeled explicitly (folded into rate constants).
* Msg5/Ptp2/Ptp3 phosphatase feedback on Fus3 is NOT modeled --
  k_deactivate_Fus3 is a fixed parameter, not induced by Module 4.
* All 8 rate constants below are PLACEHOLDERS. I could not retrieve
  Kofahl & Klipp (2004)'s fitted parameter table through web search
  (it's in the paper's PDF/supplementary SBML, not indexed as
  searchable text) -- these are order-of-magnitude estimates anchored
  to the *observed pathway activation timescale* instead (see each
  parameter's comment). Before trusting quantitative predictions,
  replace these with values transcribed directly from:
    - Kofahl B, Klipp E (2004) Yeast 21:831-850, or its SBML encoding
      BIOMD0000000032 (https://www.ebi.ac.uk/biomodels/BIOMD0000000032)
    - or your own wet-lab timecourse data.

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
# Reads Module 2's saved [Gbg](t) files; does NOT import or
# re-implement Modules 1/2. Writes its own output files for Module 4.
INTERMEDIATE_DIR = "../intermediate"
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

def load_module2_output(L_nM):
    """Load [Gbg](t) written by module2_gprotein.py for a given [L]."""
    path = os.path.join(INTERMEDIATE_DIR, f"module2_output_L{L_nM:.1f}nM.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run module1_binding.py then "
            f"module2_gprotein.py first to generate this module's "
            f"input for [L] = {L_nM} nM."
        )
    data = np.loadtxt(path)
    return data[:, 0], data[:, 3]   # t, Gbg (column 3 -- see module2 header)

# ──────────────────────────────────────────────────────────────
# PARAMETERS
# ──────────────────────────────────────────────────────────────
#
# --- Total pool sizes -------------------------------------------------
# Source: SGD (Saccharomyces Genome Database) median protein abundance,
# https://www.yeastgenome.org -- these are the unified/normalized
# values (Ho, Baryshnikova & Brown 2018, Cell Syst 6(2):192-205),
# which aggregate ~20 studies including Ghaemmaghami et al. (2003)
# Nature 425:737 (the same primary dataset Modules 1-2 cite for
# R_total/G_total). Converted molecules/cell -> nM using the same
# 42 fL yeast cell volume assumption as Modules 1-2:
#   nM = molecules_per_cell * 1e9 / (6.022e23 * 42e-15)
#      = molecules_per_cell * 0.03954
#
# Ste20: median abundance 3869 +/- 1268 molecules/cell (SGD)
#   3869 * 0.03954 ~ 153.0 nM
Ste20_total = 153.0   # nM  SGD median abundance, FINAL

# Ste11: median abundance 1533 +/- 425 molecules/cell (SGD)
#   1533 * 0.03954 ~ 60.6 nM
Ste11_total = 60.6    # nM  SGD median abundance, FINAL

# Ste7: median abundance 1466 +/- 778 molecules/cell (SGD)
#   1466 * 0.03954 ~ 58.0 nM
Ste7_total = 58.0     # nM  SGD median abundance, FINAL

# Fus3: median abundance 4800 +/- 1651 molecules/cell (SGD)
#   4800 * 0.03954 ~ 189.8 nM
Fus3_total = 189.8    # nM  SGD median abundance, FINAL

# --- Rate constants ----------------------------------------------------
# *** ALL EIGHT PLACEHOLDERS -- see MODELING STATUS docstring above ***
# Order-of-magnitude anchor: reported Fus3 phosphorylation reaches
# near-peak within roughly 15-60 min of pheromone stimulation
# (van Drogen, Stucke, Jorritsma & Peter (2001) Nat Cell Biol
# 3:1051-1059; Hilioti et al. (2008) Curr Biol 18:1700-1706 reports an
# initial activation peak within 60 min). k_activate values are scaled
# so each tier's activation rate (k_activate * typical upstream conc.)
# is of the same order as its deactivation rate, magnitude-matched to
# Module 2's k_act (4e-4 nM^-1 s^-1) and k_hyd (4e-3 s^-1) for
# consistency with the rest of the pipeline.
# TODO: replace with Kofahl & Klipp (2004) / BIOMD0000000032 values,
# or fitted wet-lab timecourse data, before trusting these numbers.
k_activate_Ste20   = 0.083   # nM^-1 s^-1  -- Kofahl and Klipp model value FINAL
k_deactivate_Ste20  = 0.017   # s^-1        -- Kofahl and Klipp model value FINAL

k_activate_Ste11   = 0.167   # nM^-1 s^-1  -- Kofahl and Klipp model value FINAL
k_deactivate_Ste11  = 0.083   # s^-1        -- Kofahl and Klipp model value FINAL

k_activate_Ste7    = 0.783   # nM^-1 s^-1  -- Kofahl and Klipp model value FINAL
k_deactivate_Ste7   = 0.083   # s^-1        -- Kofahl and Klipp model value FINAL

k_activate_Fus3    = 5.75   # nM^-1 s^-1  -- Kofahl and Klipp model value FINAL
# k_deactivate_Fus3 lumps Msg5 + Ptp2/Ptp3 phosphatase activity into a
# single fixed rate (no transcriptional feedback -- see docstring).
# Named phosphatases: Doi et al. (1994) EMBO J 13:61-70 (Msg5);
# Zhan, Deschenes & Guan (1997) Genes Dev 11:1690-1702 (Ptp2/Ptp3).
k_deactivate_Fus3   = 0.833   # s^-1        -- Kofahl and Klipp model value FINAL

# ──────────────────────────────────────────────────────────────
# ODE DEFINITION
# ──────────────────────────────────────────────────────────────
def mapk_ode(t, y, k_act20, k_deact20, k_act11, k_deact11,
             k_act7, k_deact7, k_actF, k_deactF,
             Ste20_tot, Ste11_tot, Ste7_tot, Fus3_tot, Gbg_interp):
    """
    Mass-action MAPK cascade ODEs (v1 -- see MODELING STATUS above).

    State vector:
        y[0] = Ste20_active
        y[1] = Ste11_active
        y[2] = Ste7_active
        y[3] = Fus3_active
    Inactive pools recovered from conservation (not separate states).
    """
    Ste20a, Ste11a, Ste7a, Fus3a = y

    # [Gbg] from Module 2 interpolant (continuous input)
    Gbg = max(float(Gbg_interp(t)), 0.0)   # numerical safety clamp

    dSte20a_dt = (k_act20 * Gbg    * (Ste20_tot - Ste20a) - k_deact20 * Ste20a)
    dSte11a_dt = (k_act11 * Ste20a * (Ste11_tot - Ste11a) - k_deact11 * Ste11a)
    dSte7a_dt  = (k_act7  * Ste11a * (Ste7_tot  - Ste7a)  - k_deact7  * Ste7a)
    dFus3a_dt  = (k_actF  * Ste7a  * (Fus3_tot  - Fus3a)  - k_deactF  * Fus3a)

    return [dSte20a_dt, dSte11a_dt, dSte7a_dt, dFus3a_dt]

# ──────────────────────────────────────────────────────────────
# SIMULATION
# ──────────────────────────────────────────────────────────────
t_start  = 0
t_end    = 10800        # 3 hours, matching Modules 1-2
n_points = 5000         # higher resolution than Module 2 (2000);
                        # downstream kinetics may resolve faster dynamics

# Ligand concentrations matching Modules 1-2
L_values_nM = [1e5]
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(L_values_nM)))

y0 = [0.0, 0.0, 0.0, 0.0]   # zero basal activity (unstimulated cell)

results_m3 = {}

for L in L_values_nM:
    # --- Chain: get [Gbg](t) from Module 2's saved output ---
    t_m2, Gbg_m2 = load_module2_output(L)
    Gbg_interp = interp1d(t_m2, Gbg_m2, kind='cubic', fill_value='extrapolate')

    # --- Run Module 3 ---
    t_eval = np.linspace(t_start, t_end, n_points)
    sol = solve_ivp(
        fun    = mapk_ode,
        t_span = (t_start, t_end),
        y0     = y0,
        t_eval = t_eval,
        args   = (k_activate_Ste20, k_deactivate_Ste20,
                  k_activate_Ste11, k_deactivate_Ste11,
                  k_activate_Ste7,  k_deactivate_Ste7,
                  k_activate_Fus3,  k_deactivate_Fus3,
                  Ste20_total, Ste11_total, Ste7_total, Fus3_total,
                  Gbg_interp),
        method = 'RK45',
        rtol   = 1e-8,
        atol   = 1e-10,
    )
    results_m3[L] = sol

    # --- write all four species to disk for Module 4 ---
    out_path = os.path.join(INTERMEDIATE_DIR, f"module3_output_L{L:.1f}nM.txt")
    header = (
        "Module 3 output -- MAPK cascade (mass-action v1)\n"
        f"L = {L} nM ; Ste20_total = {Ste20_total} nM ; "
        f"Ste11_total = {Ste11_total} nM ; Ste7_total = {Ste7_total} nM ; "
        f"Fus3_total = {Fus3_total} nM\n"
        "columns: time_s   Ste20_active_nM   Ste11_active_nM   "
        "Ste7_active_nM   Fus3_active_nM"
    )
    np.savetxt(out_path,
               np.column_stack([sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]]),
               header=header, fmt="%.6e")

# ──────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)

# ── Panel A: Fus3*(t) -- signal output to Module 4 ───────────
ax1 = fig.add_subplot(gs[0, 0])
for (L, sol), c in zip(results_m3.items(), colors):
    Fus3a = sol.y[3]
    ax1.plot(sol.t / 60, Fus3a, color=c, lw=2, label=f"[L] = {L} nM")
ax1.set_xlabel("Time (min)")
ax1.set_ylabel("[Fus3*] (nM)")
ax1.set_title("A.  Active Fus3 over time\n(signal input to Module 4)")
ax1.legend(fontsize=7, loc='upper left')
ax1.set_xlim(0, t_end / 60)
ax1.set_ylim(bottom=0)

# ── Panel B: All four tiers for [L] = 5 nM = K_D ─────────────
ax2 = fig.add_subplot(gs[0, 1])
L_demo = L_values_nM[0]
sol_demo = results_m3[L_demo]
t_min = sol_demo.t / 60
ax2.plot(t_min, sol_demo.y[0], lw=2, color='steelblue',  label='Ste20*')
ax2.plot(t_min, sol_demo.y[1], lw=2, color='darkorange', label='Ste11*')
ax2.plot(t_min, sol_demo.y[2], lw=2, color='forestgreen', label='Ste7*')
ax2.plot(t_min, sol_demo.y[3], lw=2, color='crimson',     label='Fus3*')
ax2.set_xlabel("Time (min)")
ax2.set_ylabel("Active concentration (nM)")
ax2.set_title(f"B.  All four tiers\n([L] = {L_demo} nM = $K_D$)")
ax2.legend(fontsize=8)
ax2.set_xlim(0, t_end / 60)
ax2.set_ylim(bottom=0)

# ── Panel C: Steady-state Fus3* vs L (dose-response) ─────────
ax3 = fig.add_subplot(gs[1, 0])
Fus3_ss = [results_m3[L].y[3][-1] for L in L_values_nM]
ax3.semilogx(L_values_nM, Fus3_ss, 'o-', color='crimson', lw=2, ms=8)
ax3.axvline(5.0, color='steelblue', ls='--', lw=1.5, label='$K_D$ = 5 nM')
ax3.set_xlabel("[L] (nM, log scale)")
ax3.set_ylabel("[Fus3*]$_{ss}$ (nM)")
ax3.set_title("C.  Steady-state [Fus3*] vs [L]\n(Module 3 dose-response)")
ax3.legend(fontsize=9)
ax3.set_ylim(bottom=0)

# ── Panel D: k_activate_Ste11 sensitivity ─────────────────────
ax4 = fig.add_subplot(gs[1, 1])
L_fixed = 5.0   # at K_D
t_m2, Gbg_m2 = load_module2_output(L_fixed)
Gbg_interp_fixed = interp1d(t_m2, Gbg_m2, kind='cubic', fill_value='extrapolate')
t_eval = np.linspace(t_start, t_end, n_points)

k_variants = {
    '0.1x (slow relay)':   k_activate_Ste11 * 0.1,
    '1x (baseline)':       k_activate_Ste11,
    '5x (fast relay)':     k_activate_Ste11 * 5,
    '20x (fast relay)':    k_activate_Ste11 * 20,
}
variant_colors = ['#d62728', '#7f7f7f', '#2ca02c', '#1f77b4']

for (label, k11), c in zip(k_variants.items(), variant_colors):
    sol_v = solve_ivp(
        mapk_ode, (t_start, t_end), y0, t_eval=t_eval,
        args=(k_activate_Ste20, k_deactivate_Ste20,
              k11, k_deactivate_Ste11,
              k_activate_Ste7, k_deactivate_Ste7,
              k_activate_Fus3, k_deactivate_Fus3,
              Ste20_total, Ste11_total, Ste7_total, Fus3_total,
              Gbg_interp_fixed),
        method='RK45', rtol=1e-8, atol=1e-10
    )
    ax4.plot(sol_v.t / 60, sol_v.y[3], lw=2, color=c, label=label)

ax4.set_xlabel("Time (min)")
ax4.set_ylabel("[Fus3*] (nM)")
ax4.set_title("D.  k$_{activate,Ste11}$ sensitivity\n"
              f"[L] = {L_fixed} nM = $K_D$")
ax4.legend(fontsize=8)
ax4.set_xlim(0, t_end / 60)
ax4.set_ylim(bottom=0)

fig.suptitle(
    "Module 3 -- MAPK Cascade (mass-action v1, explicit Ste20, no Msg5 feedback)\n"
    f"Ste20/Ste11/Ste7/Fus3 totals from SGD median abundance [PLACEHOLDER]  |  "
    "rate constants: order-of-magnitude PLACEHOLDER (see script header)",
    fontsize=10, y=1.01
)

plt.savefig("../figures/module3_mapk.png", dpi=150, bbox_inches='tight')
plt.close()
print("Figure saved.")

# ──────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ──────────────────────────────────────────────────────────────
print("\n── Steady-state summary ─────────────────────────────────────────")
print(f"{'[L] (nM)':>10} {'[Ste20*]_ss':>12} {'[Ste11*]_ss':>12} "
      f"{'[Ste7*]_ss':>12} {'[Fus3*]_ss':>12} {'% of Fus3_tot':>14}")
print("-" * 78)
for L, sol in results_m3.items():
    s20 = sol.y[0][-1]
    s11 = sol.y[1][-1]
    s7  = sol.y[2][-1]
    f3  = sol.y[3][-1]
    pct = 100 * f3 / Fus3_total
    print(f"{L:>10.1f} {s20:>12.3f} {s11:>12.3f} {s7:>12.3f} {f3:>12.3f} {pct:>13.1f}%")

print("\n── Parameter status ──────────────────────────────────────────────")
print(f"  Ste20_total = {Ste20_total} nM  <- PLACEHOLDER (SGD median abundance)")
print(f"  Ste11_total = {Ste11_total} nM  <- PLACEHOLDER (SGD median abundance)")
print(f"  Ste7_total  = {Ste7_total} nM  <- PLACEHOLDER (SGD median abundance)")
print(f"  Fus3_total  = {Fus3_total} nM  <- PLACEHOLDER (SGD median abundance)")
print("  8x rate constants          <- PLACEHOLDER, order-of-magnitude only")
print("      (timescale-anchored to van Drogen et al. 2001 / Hilioti et al. 2008;")
print("       refine against Kofahl & Klipp 2004 / BIOMD0000000032 or wet-lab data)")
print("  k_deactivate_Fus3          <- fixed constant, NO Msg5 transcriptional feedback")