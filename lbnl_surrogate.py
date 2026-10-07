"""
lbnl_surrogate.py
-----------------
Physics-based STAND-IN for the LBNL FDD "Simulated chiller plant" data set
(DOI 10.25984/1881324). It is used only to develop/test the Model A notebook
when the real CSVs are not available. It reproduces:

  * the same 77 point names (Table 2 of the LBNL inventory PDF), units (degF, GPM, W, kW, inH2O, 0-1)
  * the same 22 file names / fault types / severities (Table 4)
  * the same plant layout and control sequences (3 chillers, 3 primary + 3 condenser
    pumps, 2 secondary pumps, 3 towers, 3-way condenser bypass valve, CHW reset from
    OA dry-bulb 60-80 F -> 52-42 F, tower setpoint = WB + 8 F, loop >= 60 F,
    secondary DP setpoint 35 psi, staging at 95 % of capacity)

It is a quasi-steady model (no thermal dynamics), sampled at 15 min instead of 1 min,
with a synthetic Chicago-like weather year. Numbers it produces are NOT the LBNL
numbers - always re-run the notebook on the real files before quoting results.
"""
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------- constants
CAP_W = 1.30e6            # chiller capacity, W (~370 TR) x 3
GPM_CH = 887.0            # primary flow per chiller (10 F design dT)
GPM_CD = 1109.0           # condenser flow per chiller
K = 500.0 * 0.29307       # W per (gpm * F)
DP_SET = 35.0 * 27.6799   # 35 psi in inH2O
NTU0 = 0.887              # tower NTU at full fan speed, clean
TWR_FAN_KW = 30.0
PRI_PM_KW, CD_PM_KW, SEC_PM_KW = 15.0, 22.0, 45.0
SEC_DESIGN_GPM = 1500.0   # per secondary pump

FAULTS = {
    "ChillerPlant.csv": ("fault_free", 0),
    "ChillerPlant_coolingtower_bias_-1.csv": ("ct1_sensor_bias", -1),
    "ChillerPlant_coolingtower_bias_-2.csv": ("ct1_sensor_bias", -2),
    "ChillerPlant_coolingtower_bias_1.csv": ("ct1_sensor_bias", 1),
    "ChillerPlant_coolingtower_bias_2.csv": ("ct1_sensor_bias", 2),
    "ChillerPlant_chiller_bias_-1.csv": ("chiller1_sensor_bias", -1),
    "ChillerPlant_chiller_bias_-2.csv": ("chiller1_sensor_bias", -2),
    "ChillerPlant_chiller_bias_1.csv": ("chiller1_sensor_bias", 1),
    "ChillerPlant_chiller_bias_2.csv": ("chiller1_sensor_bias", 2),
    "ChillerPlant_secondary_chilled_water_pressure_bias_-010.csv": ("sec_dp_sensor_bias", -0.10),
    "ChillerPlant_secondary_chilled_water_pressure_bias_-020.csv": ("sec_dp_sensor_bias", -0.20),
    "ChillerPlant_secondary_chilled_water_pressure_bias_010.csv": ("sec_dp_sensor_bias", 0.10),
    "ChillerPlant_secondary_chilled_water_pressure_bias_020.csv": ("sec_dp_sensor_bias", 0.20),
    "ChillerPlant_bypass_leakage_025.csv": ("valve_leakage", 0.25),
    "ChillerPlant_bypass_leakage_050.csv": ("valve_leakage", 0.50),
    "ChillerPlant_bypass_leakage_075.csv": ("valve_leakage", 0.75),
    "ChillerPlant_bypass_stuck_050.csv": ("valve_stuck", 0.50),
    "ChillerPlant_bypass_stuck_075.csv": ("valve_stuck", 0.75),
    "ChillerPlant_coolingtower_fouling_065.csv": ("ct1_fouling", 0.65),
    "ChillerPlant_coolingtower_fouling_080.csv": ("ct1_fouling", 0.80),
    "ChillerPlant_coolingtower_fouling_095.csv": ("ct1_fouling", 0.95),
    "ChillerPlant_coolingtower_PI.csv": ("ct_pi_tuning", 1),
}


def _weather_and_load(freq_min=15, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01 01:00", "2019-12-31 23:45", freq=f"{freq_min}min")
    n = len(idx)
    doy = idx.dayofyear.values + idx.hour.values / 24
    hr = idx.hour.values + idx.minute.values / 60
    # synoptic AR(1) noise on daily scale
    days = int(np.ceil(n * freq_min / 1440)) + 2
    syn = np.zeros(days)
    for d in range(1, days):
        syn[d] = 0.8 * syn[d - 1] + rng.normal(0, 4.5)
    syn_t = np.interp(doy, np.arange(days), syn)
    oa = 50 - 24 * np.cos(2 * np.pi * (doy - 15) / 365) + 9 * np.cos(2 * np.pi * (hr - 15) / 24) + syn_t
    rh = np.clip(0.68 - 0.18 * np.cos(2 * np.pi * (hr - 15) / 24) + rng.normal(0, 0.05, n), 0.25, 0.98)
    tc = (oa - 32) / 1.8
    rhp = rh * 100
    wb_c = (tc * np.arctan(0.151977 * np.sqrt(rhp + 8.313659)) + np.arctan(tc + rhp)
            - np.arctan(rhp - 1.676331) + 0.00391838 * rhp ** 1.5 * np.arctan(0.023101 * rhp) - 4.686035)
    wb = wb_c * 1.8 + 32
    occ = ((idx.dayofweek.values < 5) & (hr >= 7) & (hr < 19)).astype(float)
    q = (0.25e6 + 0.95e6 * occ + 45000 * (oa - 55) + 30000 * np.maximum(wb - 55, 0)
         + rng.normal(0, 40000, n))
    q = np.clip(q, 0, 0.97 * 3 * CAP_W)
    q[q < 150e3] = 0.0
    oa_hourly = pd.Series(oa, idx).resample("1h").mean().reindex(idx, method="ffill").values
    return idx, oa, wb, q, oa_hourly


def _tower_out(wb, rng_F, ua, speed):
    ntu = NTU0 * ua * (0.2 + 0.8 * speed) ** 0.6
    eff = 1 - np.exp(-ntu)
    return wb + rng_F * (1 - eff) / eff


def simulate(fault="fault_free", sev=0, freq_min=15, seed=7):
    idx, oa, wb, q, oa_h = _weather_and_load(freq_min, seed)
    n = len(idx)
    import zlib
    rng = np.random.default_rng(zlib.crc32(f"{fault}|{sev}".encode()))
    on = q > 0
    nch = np.where(on, np.clip(np.ceil(q / (0.95 * CAP_W)), 1, 3), 0).astype(int)

    # ---- chilled-water setpoint reset (eq. 2)
    tset = np.clip(52 + (42 - 52) / (80 - 60) * (oa_h - 60), 42, 52)
    b_ch = 1.8 * sev if fault == "chiller1_sensor_bias" else 0.0
    tsup = np.tile(tset, (3, 1))
    tsup[0] = tset - b_ch                      # chiller 1 controls to a biased sensor
    run = np.array([nch >= i + 1 for i in range(3)])
    tsup_pri = np.where(on, (tsup * run).sum(0) / np.maximum(nch, 1), np.nan)

    # extra latent load when supply colder than setpoint
    q_tot = q * (1 + 0.03 * np.clip(tset - tsup_pri, -3, 6))
    q_tot = np.where(on, q_tot, 0)
    plr_plant = q_tot / (3 * CAP_W)

    # ---- secondary loop
    dT_sec = 6 + 5 * np.clip(plr_plant * 1.5, 0, 1)
    gpm_sec = np.where(on, q_tot / (K * dT_sec), 0)
    gpm_sec *= 1 + 0.08 * np.nan_to_num(tsup_pri - tset)   # warmer water -> valves open
    gpm_pri = nch * GPM_CH
    deficit = gpm_sec > gpm_pri
    gpm_sec = np.where(deficit, gpm_pri, gpm_sec)
    tret_sec = np.where(on, tsup_pri + q_tot / (K * np.maximum(gpm_sec, 1)), np.nan)
    tret_pri = np.where(on, tsup_pri + q_tot / (K * np.maximum(gpm_pri, 1)), np.nan)
    q_i = np.where(run, K * GPM_CH * (tret_pri - tsup), 0)
    q_i = np.clip(q_i, 0, CAP_W * 1.05)
    plr_i = q_i / CAP_W

    # secondary pumps / DP sensor bias: controller holds MEASURED dp at setpoint
    b_dp = sev if fault == "sec_dp_sensor_bias" else 0.0
    dp_act = DP_SET / (1 + b_dp)
    nsec = np.where(on, np.where(gpm_sec > 0.9 * SEC_DESIGN_GPM, 2, 1), 0)
    qpp = gpm_sec / np.maximum(nsec, 1) / SEC_DESIGN_GPM
    spd_sec = np.where(on, np.sqrt(0.4 * qpp ** 2 + 0.6 * dp_act / DP_SET), 0)
    p_sec_each = SEC_PM_KW * spd_sec ** 3

    # ---- condenser side (iterate chiller power <-> tower)
    tct_set = np.maximum(wb + 8, 60)
    b_ct = 1.8 * sev if fault == "ct1_sensor_bias" else 0.0
    ua = np.ones((3, 1)) * np.ones(n)
    if fault == "ct1_fouling":
        ua[0] = sev
    if fault == "valve_leakage":
        byp_min = 0.30 * sev
    elif fault == "valve_stuck":
        byp_min = 0.35 * sev
    else:
        byp_min = 0.0
    stuck = fault == "valve_stuck"

    p_i = q_i / 5.0
    gpm_cd_tot = nch * GPM_CD
    for _ in range(4):
        qrej = (q_i + p_i * 1000).sum(0)
        R_tot = np.where(on, qrej / (K * np.maximum(gpm_cd_tot, 1)), 0)
        f_t = np.full(n, 1 - byp_min)
        R_t = R_tot / f_t
        # controller target on the AVERAGE of logged tower sensors -> actual target
        target = tct_set - b_ct / np.maximum(nch, 1)
        lo, hi = np.zeros(n), np.ones(n)
        for _b in range(22):                       # bisection for common fan speed
            mid = (lo + hi) / 2
            outs = np.array([_tower_out(wb, R_t, ua[k], mid) for k in range(3)])
            mix = (outs * run).sum(0) / np.maximum(nch, 1)
            too_warm = mix > target
            lo = np.where(too_warm, mid, lo)
            hi = np.where(too_warm, hi, mid)
        spd = (lo + hi) / 2
        if fault == "ct_pi_tuning":
            osc = np.zeros(n)
            e = rng.normal(0, 0.16, n)
            for t in range(1, n):
                osc[t] = -0.55 * osc[t - 1] + e[t]
            spd_act = np.clip(spd + osc, 0, 1)
        else:
            spd_act = spd
        outs = np.array([_tower_out(wb, R_t, ua[k], spd_act) for k in range(3)])
        tct_mix = (outs * run).sum(0) / np.maximum(nch, 1)
        # 3-way valve keeps loop supply >= 60 F (unless stuck)
        if not stuck:
            need = tct_mix + (1 - f_t) / f_t * R_tot < 60
            f_need = R_tot / np.maximum(60 - tct_mix + R_tot, 1e-6)
            f_t = np.where(need, np.clip(f_need, 0.05, 1 - byp_min), f_t)
        tcd_sup = tct_mix + (1 - f_t) / f_t * R_tot
        tcd_ret = tcd_sup + R_tot
        # chiller power from lift and part load
        tcd_leave_i = tcd_sup + np.where(run, (q_i + p_i * 1000) / (K * GPM_CD), 0)
        tcond = tcd_leave_i + 3 + 459.67
        tevap = tsup - 4 + 459.67
        eta = 0.56 * (1 - 0.9 * (plr_i - 0.6) ** 2)
        cop = np.clip(eta * tevap / np.maximum(tcond - tevap, 5), 1.5, 9.5)
        p_i = np.where(run, q_i / cop / 1000 + 12, 0)      # kW
    byp = 1 - f_t
    p_twr = np.where(run, TWR_FAN_KW * spd_act ** 3, 0)

    # ---------------------------------------------------------- assemble points
    def nz(x, sd):
        return x + rng.normal(0, sd, n)

    def nzp(x, frac):
        return x * (1 + rng.normal(0, frac, n))

    d = {}
    amb_w = np.round(oa * 0.4 + 30, 2)          # idle-loop temperature drifts towards ambient
    d["CWL_SEC_LOAD"] = np.where(on, nzp(K * gpm_sec * (tret_sec - tsup_pri), 0.01), 0)
    d["CWL_SEC_RW_TEMP"] = np.where(on, nz(tret_sec, 0.1), amb_w)
    d["CWL_SEC_SW_TEMP"] = np.where(on, nz(tsup_pri, 0.1), amb_w)
    d["CWL_SEC_CW_FLOW"] = nzp(gpm_sec, 0.005)
    d["TWV_CTRL"] = np.where(on, byp, 0)
    d["CT_SW_TEMPSPT"] = tct_set
    d["CDWL_SW_TEMP"] = np.where(on, nz(tcd_sup, 0.1), amb_w)
    d["CDWL_RW_TEMP"] = np.where(on, nz(tcd_ret, 0.1), amb_w)
    d["CDWL_CW_FLOW"] = nzp(gpm_cd_tot, 0.005)
    d["CWL_PRI_RW_TEMP"] = np.where(on, nz(tret_pri, 0.1), amb_w)
    d["CWL_PRI_SW_TEMP"] = np.where(on, nz(tsup_pri, 0.1), amb_w)
    d["CWL_PRI_CW_FLOW"] = nzp(gpm_pri, 0.005)
    d["CWL_PRI_SW_TEMPSPT"] = tset
    d["OA_TEMP"] = nz(oa, 0.2)
    d["OA_TEMP_WB"] = nz(wb, 0.2)
    d["CWL_SEC_DPSPTS"] = np.full(n, DP_SET)
    d["CWL_SEC_DP"] = np.where(on, nzp(dp_act * (1 + b_dp), 0.01), 0)
    for k in range(3):
        j = str(k + 1)
        r = run[k]
        log_ct = outs[k] + (b_ct if k == 0 else 0)
        d["CT_STA" + j] = r.astype(int)
        d["CT_FLOW" + j] = np.where(r, nzp(gpm_cd_tot * f_t / np.maximum(nch, 1), 0.005), 0)
        d["CT_RW_TEMP" + j] = np.where(r, nz(tcd_ret, 0.1), amb_w)
        d["CT_SW_TEMP" + j] = np.where(r, nz(log_ct, 0.1), amb_w)
        d["CT_FAN_SPD" + j] = np.where(r, spd_act, 0)
        d["CT_FAN_SPD_CTRL" + j] = np.where(r, spd, 0)
        d["CT_POW" + j] = np.where(r, nzp(p_twr[k], 0.005), 0)
    for k in range(3):
        j = str(k + 1)
        r = run[k]
        log_sw = tsup[k] + (b_ch if k == 0 else 0)
        d["CHL_STA" + j] = r.astype(int)
        d["CHL_COMP_SPD_CTRL" + j] = np.where(r, plr_i[k], 0)
        d["CHL_CW_FLOW" + j] = np.where(r, nzp(np.full(n, GPM_CH), 0.005), 0)
        d["CHL_CD_FLOW" + j] = np.where(r, nzp(np.full(n, GPM_CD), 0.005), 0)
        d["CHL_RW_TEMP" + j] = np.where(r, nz(tret_pri, 0.1), amb_w)
        d["CHL_SW_TEMP" + j] = np.where(r, nz(log_sw, 0.1), amb_w)
        d["CHL_SWCD_TEMP" + j] = np.where(r, nz(tcd_leave_i[k], 0.1), amb_w)
        d["CHL_RWCD_TEMP" + j] = np.where(r, nz(tcd_sup, 0.1), amb_w)
        d["CHL_POW" + j] = np.where(r, nzp(p_i[k], 0.005), 0)
    for k in range(3):
        d["CDWL_PM_POW" + str(k + 1)] = np.where(run[k], nzp(np.full(n, CD_PM_KW), 0.005), 0)
    for k in range(3):
        d["CWL_PRI_PM_POW" + str(k + 1)] = np.where(run[k], nzp(np.full(n, PRI_PM_KW), 0.005), 0)
    for k in range(2):
        r = nsec >= k + 1
        d["CWL_SEC_PM_POW" + str(k + 1)] = np.where(r, nzp(p_sec_each, 0.005), 0)
        d["CWL_SEC_PM_SPD" + str(k + 1)] = np.where(r, spd_sec, 0)
        d["CWL_SEC_PM_STA" + str(k + 1)] = r.astype(int)
    df = pd.DataFrame(d, index=idx)
    df.index.name = "Datetime"
    return df.round(4)


def build_all(out_dir="data_surrogate", freq_min=15):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for fname, (fault, sev) in FAULTS.items():
        simulate(fault, sev, freq_min).to_csv(out / fname)
    return out


if __name__ == "__main__":
    p = build_all()
    print("wrote", len(list(p.glob("*.csv"))), "files to", p)
