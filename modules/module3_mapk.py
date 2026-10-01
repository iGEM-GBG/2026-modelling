"""
Module 3: MAPK Cascade (Ste20 -> Ste11 -> Ste7 -> Fus3)
==========================================================
Models the four-tier phosphorylation relay downstream of the G-protein
cycle, using a v1 MASS-ACTION approximation (no saturation kinetics
yet -- see v2 for the Goldbeter-Koshland saturating version):

    G_betagamma + Ste20   --k_activate_Ste20-->  Ste20*
    Ste20*      + Ste11   --k_activate_Ste11-->  Ste11*
    Ste11*      + Ste7    --k_activate_Ste7-->   Ste7*
    Ste7*       + Fus3    --k_activate_Fus3-->   Fus3*

    Ste20* --k_deactivate_Ste20--> Ste20   (etc. for each tier)

Each tier is a 2-state lumped switch (active / inactive), NOT a
3-state distributive dual-phosphorylation model.

Conservation laws (one per tier):
    Ste20_total = Ste20 + Ste20*
    Ste11_total = Ste11 + Ste11*
    Ste7_total  = Ste7  + Ste7*
    Fus3_total  = Fus3  + Fus3*

State vector: y = [Ste20_active, Ste11_active, Ste7_active, Fus3_active]

Output to Module 4: [Fus3*](t)  -->  drives Ste12 activation

Chain: Module 2 [Gbetagamma](t) --> text file --> interp1d --> input here

MODELING STATUS
----------------
* Ste5 scaffold is NOT modeled explicitly (folded into rate constants).
  Kofahl & Klipp's own rate constants (below) were fitted inside their
  scaffold-complex model; reusing them in this lumped bimolecular form
  is an anchor, not a structural match -- treat absolute dynamics with
  that in mind.
* Msg5/Ptp2/Ptp3 phosphatase feedback on Fus3 is NOT modeled --
  k_deactivate_Fus3 is a fixed parameter, not induced by Module 4.

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
# Total pool sizes -- SGD median protein abundance (Ho, Baryshnikova &
# Brown 2018, Cell Syst 6(2):192-205), converted molecules/cell -> nM
# at 42 fL cell volume (nM = molecules_per_cell * 0.03954).
Ste20_total = 153.0   # nM -- 3869 molecules/cell
Ste11_total = 60.6    # nM -- 1533 molecules/cell
Ste7_total  = 58.0    # nM -- 1466 molecules/cell
Fus3_total  = 189.8   # nM -- 4800 molecules/cell

# Rate constants -- Kofahl & Klipp (2004) Yeast 21:831-850, Table 2
# model values, converted min^-1 -> s^-1 (divide by 60).
k_activate_Ste20   = 0.083   # nM^-1 s^-1
k_deactivate_Ste20 = 0.017   # s^-1

k_activate_Ste11   = 0.167   # nM^-1 s^-1
k_deactivate_Ste11 = 0.083   # s^-1

k_activate_Ste7    = 0.783   # nM^-1 s^-1
k_deactivate_Ste7  = 0.083   # s^-1

k_activate_Fus3    = 5.75    # nM^-1 s^-1
# k_deactivate_Fus3 lumps Msg5 + Ptp2/Ptp3 phosphatase activity into a
# single fixed rate (no transcriptional feedback). Named phosphatases:
# Doi et al. (1994) EMBO J 13:61-70 (Msg5); Zhan, Deschenes & Guan
# (1997) Genes Dev 11:1690-1702 (Ptp2/Ptp3).
k_deactivate_Fus3  = 0.833   # s^-1

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
t_start  = 0
t_end    = 10800        # 3 hours, matching Modules 1-2

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
    t_eval = dense_early_t_eval(t_start, t_end)
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

L_demo = L_values_nM[0]

# ──────────────────────────────────────────────────────────────
# PLOTTING -- each panel saved as its own PDF
# ──────────────────────────────────────────────────────────────

# ── A: Fus3*(t) -- signal output to Module 4 ─────────────────
figA, axA = plt.subplots(figsize=(6.5, 5))
for (L, sol), c in zip(results_m3.items(), colors):
    axA.plot(sol.t / 60, sol.y[3], color=c, lw=2, label=f"[L] = {L:.0f} nM")
axA.set_xlabel("Time (min)")
axA.set_ylabel("[Fus3*] (nM)")
axA.set_title("Active Fus3 over time\n(signal input to Module 4)")
axA.legend(fontsize=9, loc='upper left')
axA.set_xlim(0, 10)   # zoomed: real kinetics settle within ~2 min (see design discussion)
axA.set_ylim(bottom=0)
figA.tight_layout()
figA.savefig(os.path.join(FIGURE_DIR, "module3_A_Fus3_timecourse.pdf"))
plt.close(figA)

# ── B: All four tiers at the simulated dose ───────────────────
figB, axB = plt.subplots(figsize=(6.5, 5))
sol_demo = results_m3[L_demo]
t_min = sol_demo.t / 60
axB.plot(t_min, sol_demo.y[0], lw=2, color='steelblue',  label='Ste20*')
axB.plot(t_min, sol_demo.y[1], lw=2, color='darkorange', label='Ste11*')
axB.plot(t_min, sol_demo.y[2], lw=2, color='forestgreen', label='Ste7*')
axB.plot(t_min, sol_demo.y[3], lw=2, color='crimson',     label='Fus3*')
axB.set_xlabel("Time (min)")
axB.set_ylabel("Active concentration (nM)")
axB.set_title(f"All four tiers\n([L] = {L_demo:.0f} nM, wet-lab dose)")
axB.legend(fontsize=9)
axB.set_xlim(0, 10)   # zoomed: real kinetics settle within ~2 min (see design discussion)
axB.set_ylim(bottom=0)
figB.tight_layout()
figB.savefig(os.path.join(FIGURE_DIR, "module3_B_all_tiers.pdf"))
plt.close(figB)

# ── C: k_activate_Ste11 sensitivity ───────────────────────────
figC, axC = plt.subplots(figsize=(6.5, 5))
t_m2, Gbg_m2 = load_module2_output(L_demo)
Gbg_interp_fixed = interp1d(t_m2, Gbg_m2, kind='cubic', fill_value='extrapolate')
t_eval = dense_early_t_eval(t_start, t_end)

k_variants = {
    '0.1x (slow relay)':  k_activate_Ste11 * 0.1,
    '1x (baseline)':      k_activate_Ste11,
    '5x (fast relay)':    k_activate_Ste11 * 5,
    '20x (fast relay)':   k_activate_Ste11 * 20,
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
        method='LSODA', rtol=1e-8, atol=1e-10
    )
    axC.plot(sol_v.t / 60, sol_v.y[3], lw=2, color=c, label=label)

axC.set_xlabel("Time (min)")
axC.set_ylabel("[Fus3*] (nM)")
axC.set_title(f"k$_{{activate,Ste11}}$ sensitivity\n[L] = {L_demo:.0f} nM")
axC.legend(fontsize=9)
axC.set_xlim(0, 10)   # zoomed: real kinetics settle within ~2 min (see design discussion)
axC.set_ylim(bottom=0)
figC.tight_layout()
figC.savefig(os.path.join(FIGURE_DIR, "module3_C_kactivate_sensitivity.pdf"))
plt.close(figC)

print("Figures saved: module3_{A_Fus3_timecourse,B_all_tiers,"
      "C_kactivate_sensitivity}.pdf")

# ──────────────────────────────────────────────────────────────
# STEADY-STATE SUMMARY
# ──────────────────────────────────────────────────────────────
print("\n-- Steady-state summary ----------------------------------------")
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