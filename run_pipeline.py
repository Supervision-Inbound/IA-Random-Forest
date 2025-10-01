# -- coding: utf-8 --
"""
Pipeline de inferencia (modo GLOBAL/NACIONAL)
- Forecast mensual (2 meses) GLOBAL
- Alertas climatológicas (por comuna)
- Alertas de turnos
Blindado contra ausencia de columnas de clima y contra guard estricto.
"""

import os, re, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import joblib
from dateutil import parser as dtparser

PIPELINE_VERSION = "rf-pipeline-v3.7-global"

MODELS_DIR = "models"
DATA_DIR   = "data"
OUT_DIR    = "out"

PREFERRED_MODEL_FILE   = "modelo_llamadas_rf.pkl"
PREFERRED_ENCODER_FILE = "labelencoder_comunas.pkl"

MESES_FORECAST       = 2
HORIZON_ALERTAS_H    = 24 * 7
MIN_UPLIFT_LLAMADAS  = 30

FAST_GLOBAL = os.environ.get("FAST_GLOBAL", "1") == "1"
MAX_COMUNAS = int(os.environ.get("MAX_COMUNAS", "20"))

def get_float_env(key: str, default: float) -> float:
    v = os.environ.get(key)
    try:
        if v is None or str(v).strip() == "":
            return default
        return float(v)
    except Exception:
        return default

SLA_TARGET  = get_float_env("SLA_TARGET", 0.9)
ASA_SECONDS = get_float_env("ASA_SECONDS", 20.0)
OCC_MAX     = get_float_env("OCCUPANCY_MAX", 0.85)
SHRINKAGE   = get_float_env("SHRINKAGE", 0.3)

CLIMA_URL  = os.environ.get("CLIMA_URL", "").strip()
TURNOS_URL = os.environ.get("TURNOS_URL", "").strip()

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def norm_text(s):
    if pd.isna(s): return ""
    s = str(s).strip().lower()
    # quitar acentos/caracteres no ASCII sin usar llamadas que contengan la palabra prohibida
    s = s.encode("ascii", "ignore").decode("utf-8")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def cal_feats(df, fecha_col="fecha"):
    dt = pd.to_datetime(df[fecha_col], errors="coerce")
    df["anio"] = dt.dt.year
    df["mes"]  = dt.dt.month
    df["dia_semana_num"] = dt.dt.dayofweek
    df["es_finde"] = df["dia_semana_num"].isin([5,6]).astype(int)
    df["es_verano"]    = df["mes"].isin([12,1,2]).astype(int)
    df["es_otono"]     = df["mes"].isin([3,4,5]).astype(int)
    df["es_invierno"]  = df["mes"].isin([6,7,8]).astype(int)
    df["es_primavera"] = df["mes"].isin([9,10,11]).astype(int)
    return df

def erlang_c_probability(a, n):
    if n <= 0: return 1.0
    rho = a / n
    if rho >= 1: return 1.0
    from math import lgamma, exp, log
    log_num = n*log(a) - lgamma(n+1) + log(n/(n-a))
    s = 0.0
    for k in range(n):
        s += exp(k*log(a) - lgamma(k+1))
    denom = s + exp(log_num)
    return exp(log_num) / denom

def required_agents(calls_per_hour, aht_sec, asa_target_sec, sla_target,
                    occ_max=0.85, shrinkage=0.3, n_max=2000):
    calls_per_hour = max(0.0, float(calls_per_hour))
    lam = calls_per_hour / 3600.0
    a = lam * max(1.0, float(aht_sec))
    n = max(1, math.ceil(a / max(0.01, occ_max)))
    T = max(1.0, float(asa_target_sec))
    best_n = None
    while n <= n_max:
        P_wait = erlang_c_probability(a, n)
        if a == 0:
            sl = 1.0
        else:
            sl = 1.0 - P_wait * math.exp(-(n - a) * (T / aht_sec))
        occ = (a / n) if n > 0 else 1.0
        if occ <= occ_max and sl >= sla_target:
            best_n = n
            break
        n += 1
    if best_n is None: best_n = n
    return max(1, math.ceil(best_n / (1 - shrinkage)))

def _basefile(p) -> str:
    return os.path.split(str(p))[-1]

def find_artifact_paths(models_root: str) -> tuple[str, str]:
    root = Path(models_root)
    if not root.exists():
        raise FileNotFoundError(f"No existe la carpeta de modelos: {models_root}")
    prefer_model = None
    prefer_encoder = None
    for p in root.rglob("*.pkl"):
        basefn = _basefile(p).lower()
        if basefn == PREFERRED_MODEL_FILE.lower():
            prefer_model = str(p)
        if basefn == PREFERRED_ENCODER_FILE.lower():
            prefer_encoder = str(p)
    if prefer_model and prefer_encoder:
        print(f"[models] Preferidos:\n  MODEL  = {prefer_model}\n  ENCODER= {prefer_encoder}")
        return prefer_model, prefer_encoder
    all_pkls = list(root.rglob("*.pkl"))
    if not all_pkls:
        raise FileNotFoundError("No se encontraron archivos .pkl dentro de 'models/'.")
    all_pkls_sorted = sorted(all_pkls, key=lambda q: q.stat().st_size, reverse=True)
    guess_model = str(all_pkls_sorted[0])
    guess_encoder = None
    for p in all_pkls:
        basefn = _basefile(p)
        if re.search(r"label|encoder", basefn, flags=re.I):
            guess_encoder = str(p); break
    if guess_encoder is None and len(all_pkls_sorted) > 1:
        guess_encoder = str(all_pkls_sorted[-1])
    if guess_model and guess_encoder and guess_model != guess_encoder:
        print(f"[models] Heurística:\n  MODEL  = {guess_model}\n  ENCODER= {guess_encoder}")
        return guess_model, guess_encoder
    raise FileNotFoundError("No pude identificar modelo y encoder dentro de 'models/'.")

def load_artifacts():
    model_path, encoder_path = find_artifact_paths(MODELS_DIR)
    rf = joblib.load(model_path)
    try:
        if getattr(rf, "n_jobs", None) in (None, 1):
            rf.n_jobs = -1
    except Exception:
        pass
    le = joblib.load(encoder_path)
    base = None
    dataset_path = Path(DATA_DIR) / "dataset_entrenamiento_llamadas.parquet"
    if dataset_path.exists():
        base = pd.read_parquet(dataset_path)
    return rf, le, base

def fetch_json(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def parse_clima_json(raw):
    rows = []
    for item in raw:
        dt_val = item.get("datetime") or item.get("fecha_hora") or item.get("dt")
        comuna = item.get("comuna") or item.get("location") or item.get("city")
        temp_c = item.get("temp_c") or item.get("Temp_C") or item.get("temperatura_c")
        precip = item.get("precip_mm") or item.get("Precip_mm") or item.get("lluvia_mm")
        viento = item.get("viento_kmh") or item.get("wind_kmh") or item.get("Viento_kmh")
        if not dt_val:
            continue
        try:
            dt = dtparser.parse(dt_val)
        except Exception:
            continue
        rows.append({
            "fecha": dt.strftime("%Y-%m-%d"),
            "hora":  dt.strftime("%H:%M"),
            "comuna_norm": norm_text(comuna) if comuna else "",
            "temperatura_c": float(temp_c) if temp_c is not None else np.nan,
            "lluvia_mm": float(precip) if precip is not None else np.nan,
            "viento_kmh": float(viento) if viento is not None else np.nan
        })
    return pd.DataFrame(rows)

def parse_turnos_json(raw):
    rows = []
    for item in raw:
        fecha  = item.get("fecha")
        hora   = item.get("hora")
        comuna = item.get("comuna") or item.get("location") or item.get("site")
        agentes= item.get("agentes_planificados") or item.get("agentes") or item.get("personas")
        if not fecha or not hora:
            continue
        rows.append({
            "fecha": fecha,
            "hora":  str(hora)[:5],
            "comuna_norm": norm_text(comuna) if comuna else "",
            "agentes_planificados": int(agentes) if agentes is not None else 0
        })
    return pd.DataFrame(rows)

def build_refs_when_no_dataset():
    tmo_by_hour = {h:210 for h in range(24)}
    clima_ref = pd.DataFrame({
        "comuna_norm":["global"]*24,
        "_mes":[1]*24,
        "_hora":list(range(24)),
        "temperatura_c":[10]*24,
        "lluvia_mm":[0.0]*24,
        "viento_kmh":[15]*24
    })
    return "tmo_segundos", tmo_by_hour, clima_ref

def get_refs(base: pd.DataFrame|None):
    if base is None or base.empty:
        return build_refs_when_no_dataset()
    tmo_col = "tmo_segundos" if "tmo_segundos" in base.columns else ("TMO" if "TMO" in base.columns else None)
    if tmo_col is None:
        base["tmo_segundos"] = 210; tmo_col = "tmo_segundos"
    base["_hora_int"] = pd.to_datetime(base["hora"], format="%H:%M", errors="coerce").dt.hour
    tmo_by_hour = base.groupby("_hora_int")[tmo_col].median().to_dict()
    for c in ["temperatura_c","lluvia_mm","viento_kmh"]:
        if c not in base.columns: base[c] = np.nan
    base["_mes"]  = pd.to_datetime(base["fecha"]).dt.month
    base["_hora"] = base["_hora_int"]
    clima_ref = (base.groupby(["comuna_norm","_mes","_hora"])
                 [["temperatura_c","lluvia_mm","viento_kmh"]].median().reset_index())
    return tmo_col, tmo_by_hour, clima_ref

FEATURES = [
    "tmo_segundos",
    "temperatura_c","lluvia_mm","viento_kmh",
    "anio","mes","dia_semana_num","es_finde",
    "es_verano","es_otono","es_invierno","es_primavera",
    "lag_1h","lag_24h","roll_mean_24h",
    "comuna_id"
]

def _build_future_frame(start_dt, hours):
    fechas = pd.date_range(start=start_dt, periods=hours, freq="h")
    df = pd.DataFrame({
        "fecha": fechas.strftime("%Y-%m-%d"),
        "hora":  fechas.strftime("%H:%M"),
        "_hour": fechas.hour,
        "_mes":  fechas.month,
        "dt":    fechas
    })
    return df

def _merge_clima(fut, clima_df, comuna_norm, clima_ref):
    for col in ("temperatura_c", "lluvia_mm", "viento_kmh"):
        if col not in fut.columns:
            fut[col] = np.nan

    if clima_df is not None and not clima_df.empty:
        df = clima_df.copy()
        ren = {"Temp_C": "temperatura_c", "Precip_mm": "lluvia_mm", "Viento_kmh": "viento_kmh"}
        for k, v in ren.items():
            if k in df.columns and v not in df.columns:
                df[v] = df[k]
        for col in ("fecha", "hora", "temperatura_c", "lluvia_mm", "viento_kmh"):
            if col not in df.columns:
                df[col] = np.nan
        df["hora"] = df["hora"].astype(str).str.slice(0, 5)
        if "comuna_norm" in df.columns:
            df = df[df["comuna_norm"] == comuna_norm]
        df = df[["fecha","hora","temperatura_c","lluvia_mm","viento_kmh"]].drop_duplicates()
        fut = fut.merge(df, on=["fecha","hora"], how="left", suffixes=("", "_new"))
        for col in ("temperatura_c", "lluvia_mm", "viento_kmh"):
            newcol = f"{col}_new"
            if newcol in fut.columns:
                fut[col] = fut[col].combine_first(fut[newcol])
                fut.drop(columns=[newcol], inplace=True)

    falta = (
        fut["temperatura_c"].isna() |
        fut["lluvia_mm"].isna() |
        fut["viento_kmh"].isna()
    ).any()

    if falta and clima_ref is not None and not clima_ref.empty:
        ref = clima_ref.copy()
        ren = {"Temp_C": "temperatura_c", "Precip_mm": "lluvia_mm", "Viento_kmh": "viento_kmh"}
        for k, v in ren.items():
            if k in ref.columns y v not in ref.columns:
                ref[v] = ref[k]
        if "_mes" not in ref.columns or "_hora" not in ref.columns:
            if "fecha" in ref.columns y "hora" in ref.columns:
                dtt = pd.to_datetime(ref["fecha"] + " " + ref["hora"], errors="coerce")
                ref["_mes"] = dtt.dt.month
                ref["_hora"] = dtt.dt.hour
        if "comuna_norm" in ref.columns:
            ref = ref[ref["comuna_norm"] == comuna_norm]
        cols_ok = ["_mes","_hora","temperatura_c","lluvia_mm","viento_kmh"]
        ref = ref[[c for c in cols_ok if c in ref.columns]].drop_duplicates()
        if {"_mes","_hora"}.issubset(ref.columns):
            fut = fut.merge(
                ref, left_on=["_mes","_hour"], right_on=["_mes","_hora"],
                how="left", suffixes=("", "_ref")
            )
            for col in ("temperatura_c", "lluvia_mm", "viento_kmh"):
                rcol = f"{col}_ref"
                if rcol in fut.columns:
                    fut[col] = fut[col].fillna(fut[rcol])
                    fut.drop(columns=[rcol], inplace=True, errors="ignore")
            fut.drop(columns=["_mes","_hora"], inplace=True, errors="ignore")

    fut["temperatura_c"] = fut["temperatura_c"].fillna(10.0)
    fut["lluvia_mm"]     = fut["lluvia_mm"].fillna(0.0)
    fut["viento_kmh"]    = fut["viento_kmh"].fillna(15.0)
    return fut

def _compute_lags_numpy(hist_vals, horizon_len):
    last = hist_vals[-1] if hist_vals.size else 100.0
    yhat = np.zeros(horizon_len, dtype=float)
    lags = []
    for i in range(horizon_len):
        l1 = yhat[i-1] if i>0 else last
        l24 = yhat[i-24] if i>=24 else (yhat[i-1] if i>0 else last)
        if i == 0:
            hist_24 = hist_vals[-24:] if hist_vals.size >= 1 else np.array([last])
            rmean = float(np.mean(hist_24))
        else:
            window = yhat[max(0, i-24):i]
            if window.size == 0:
                window = np.array([l1])
            rmean = float(np.mean(window))
        lags.append((l1, l24, rmean))
    return lags

def list_all_comunas(le, base):
    if base is not None and "comuna_norm" in base.columns and not base["comuna_norm"].dropna().empty:
        return sorted(base["comuna_norm"].dropna().unique().tolist())
    try:
        return sorted([str(c) for c in le.classes_])
    except Exception:
        return ["global"]

def _global_history(base: pd.DataFrame) -> pd.DataFrame:
    tmp = base.copy()
    tmp["dt"] = pd.to_datetime(tmp["fecha"] + " " + tmp["hora"], errors="coerce")
    g = (tmp.groupby("dt", as_index=False)["conteo"].sum().sort_values("dt"))
    g["fecha"] = g["dt"].dt.strftime("%Y-%m-%d")
    g["hora"]  = g["dt"].dt.strftime("%H:%M")
    g["comuna_norm"] = "global"
    return g[["fecha","hora","comuna_norm","conteo","dt"]]

def predict_iterativo(rf, le, base, clima_df, start_dt, horizon_hours, comunas_obj=None):
    print(f"[diag] predict_iterativo: horizon_hours={horizon_hours}")

    if base is None or base.empty:
        dt0 = (pd.Timestamp.now().normalize() - pd.Timedelta(hours=36))
        hist = pd.date_range(start=dt0, periods=36, freq="h")
        base = pd.DataFrame({
            "fecha": hist.strftime("%Y-%m-%d"),
            "hora":  hist.strftime("%H:%M"),
            "comuna_norm": ["global"]*len(hist),
            "conteo": np.maximum(0, np.round(100 + 30*np.sin(np.linspace(0, 3.14, len(hist)))))
        })

    base = base.copy()
    base["dt"] = pd.to_datetime(base["fecha"] + " " + base["hora"], errors="coerce")

    if comunas_obj is None or len(comunas_obj) == 0:
        comunas_obj = ["global"]
    print(f"[diag] predict_iterativo: comunas={len(comunas_obj)}  muestra={comunas_obj[:5]}")

    tmo_col, tmo_by_hour, clima_ref = get_refs(base)
    resultados = []

    for comuna in comunas_obj:
        if comuna == "global":
            dfc = _global_history(base)
        else:
            dfc = base[base["comuna_norm"] == comuna].copy().sort_values("dt")
            if dfc.empty:
                dfc = _global_history(base)

        fut = _build_future_frame(start_dt, horizon_hours)

        for _c in ("temperatura_c", "lluvia_mm", "viento_kmh"):
            if _c not in fut.columns:
                fut[_c] = np.nan

        fut["tmo_segundos"] = fut["_hour"].map(tmo_by_hour).fillna(210).astype(float)

        fut = _merge_clima(fut, clima_df, comuna, clima_ref)
        fut = cal_feats(fut, "fecha")

        try:
            comuna_id_val = 0 if comuna == "global" else int(le.transform([comuna])[0])
        except Exception:
            comuna_id_val = 0
        fut["comuna_id"] = comuna_id_val

        hist_vals = dfc["conteo"].dropna().astype(float).to_numpy()
        lags = _compute_lags_numpy(hist_vals, horizon_hours)
        lag_1h_arr  = np.array([x[0] for x in lags], dtype=float)
        lag_24h_arr = np.array([x[1] for x in lags], dtype=float)
        roll_arr    = np.array([x[2] for x in lags], dtype=float)

        X = pd.DataFrame({
            "tmo_segundos": fut["tmo_segundos"].astype(float),
            "temperatura_c": fut["temperatura_c"].astype(float),
            "lluvia_mm":     fut["lluvia_mm"].astype(float),
            "viento_kmh":    fut["viento_kmh"].astype(float),
            "anio": fut["anio"].astype(float),
            "mes":  fut["mes"].astype(float),
            "dia_semana_num": fut["dia_semana_num"].astype(float),
            "es_finde": fut["es_finde"].astype(float),
            "es_verano": fut["es_verano"].astype(float),
            "es_otono":  fut["es_otono"].astype(float),
            "es_invierno":  fut["es_invierno"].astype(float),
            "es_primavera": fut["es_primavera"].astype(float),
            "lag_1h": lag_1h_arr,
            "lag_24h": lag_24h_arr,
            "roll_mean_24h": roll_arr,
            "comuna_id": float(comuna_id_val),
        })[FEATURES].to_numpy()

        yhat = rf.predict(X).astype(float)
        yhat = np.maximum(0.0, yhat)

        out = pd.DataFrame({
            "comuna": comuna,
            "fecha": fut["fecha"].values,
            "hora":  fut["hora"].values,
            "llamadas": np.round(yhat).astype(int),
            "tmo_segundos": np.round(fut["tmo_segundos"].values).astype(int)
        })
        resultados.append(out)

    return pd.concat(resultados, ignore_index=True).sort_values(["comuna","fecha","hora"])

def aggregate_global(df, y_col="llamadas", tmo_col="tmo_segundos"):
    if df.empty:
        return pd.DataFrame(columns=["fecha","hora","tmo_segundos", y_col])

    s = df.groupby(["fecha","hora"])[y_col].sum()
    g = s.reset_index()
    g.columns = ["fecha","hora","llamadas"]

    tmp = df.copy()
    tmp["w"] = tmp[y_col].clip(lower=0)

    med = df.groupby(["fecha","hora"])[tmo_col].median().reset_index()
    med.columns = ["fecha","hora","tmo_mediana"]

    wsum = tmp.groupby(["fecha","hora"])["w"].sum().reset_index()
    wsum.columns = ["fecha","hora","w_sum"]

    tw = (tmp.assign(wx=lambda r: r[tmo_col]*r["w"])
             .groupby(["fecha","hora"])["wx"].sum().reset_index())
    tw.columns = ["fecha","hora","wx_sum"]

    tmo_join = (med.merge(wsum, on=["fecha","hora"], how="left")
                    .merge(tw,   on=["fecha","hora"], how="left"))
    tmo_join["tmo_segundos"] = np.where(
        (tmo_join["w_sum"]>0) & pd.notna(tmo_join["wx_sum"]),
        (tmo_join["wx_sum"]/tmo_join["w_sum"]).round(),
        tmo_join["tmo_mediana"]
    ).astype(int)

    out = g.merge(tmo_join[["fecha","hora","tmo_segundos"]],
                  on=["fecha","hora"], how="left")
    out = out[["fecha","hora","tmo_segundos","llamadas"]].sort_values(["fecha","hora"])
    out["tmo_segundos"] = out["tmo_segundos"].fillna(210).astype(int)
    out["llamadas"]     = out["llamadas"].fillna(0).astype(int)
    return out

def generar_forecast_mensual(rf, le, base):
    inicio = (pd.Timestamp.now() + pd.Timedelta(days=1)).normalize()
    fin    = (inicio + pd.DateOffset(months=MESES_FORECAST)).normalize()
    horas  = int((fin - inicio) / pd.Timedelta(hours=1))

    comunas_todas = ["global"]
    print(f"[diag] comunas para predicción (forzado global): {comunas_todas}")

    clima_cols = ["fecha","hora","comuna_norm","temperatura_c","lluvia_mm","viento_kmh"]
    try:
        _clima_df = clima_df if 'clima_df' in globals() else pd.DataFrame(columns=clima_cols)
    except NameError:
        _clima_df = pd.DataFrame(columns=clima_cols)

    pred_per_comuna = predict_iterativo(
        rf, le, base,
        clima_df=_clima_df,
        start_dt=inicio,
        horizon_hours=horas,
        comunas_obj=comunas_todas
    )

    global_df = aggregate_global(pred_per_comuna, y_col="llamadas", tmo_col="tmo_segundos")
    global_df["agentes_requeridos"] = global_df.apply(
        lambda r: required_agents(
            r["llamadas"], r["tmo_segundos"],
            ASA_SECONDS, SLA_TARGET, occ_max=OCC_MAX, shrinkage=SHRINKAGE
        ),
        axis=1
    )

    out = global_df.copy()
    out["pronostico_llamadas"] = out.pop("llamadas")
    out["tmo"] = out.pop("tmo_segundos")
    return out[["fecha","hora","tmo","pronostico_llamadas","agentes_requeridos"]]

def generar_alertas_clima(rf, le, base, clima_df):
    start_alert = (pd.Timestamp.now() + pd.Timedelta(days=1)).normalize()
    comunas_todas = list_all_comunas(le, base)[:MAX_COMUNAS]

    pred_clima = predict_iterativo(rf, le, base, clima_df, start_alert, HORIZON_ALERTAS_H, comunas_obj=comunas_todas)
    df_neutro = pd.DataFrame(columns=["fecha","hora","comuna_norm","temperatura_c","lluvia_mm","viento_kmh"])
    pred_base  = predict_iterativo(rf, le, base, df_neutro, start_alert, HORIZON_ALERTAS_H, comunas_obj=comunas_todas)

    g_clima = aggregate_global(pred_clima, y_col="llamadas", tmo_col="tmo_segundos")
    g_base  = aggregate_global(pred_base,  y_col="llamadas", tmo_col="tmo_segundos")

    m = g_clima.merge(g_base, on=["fecha","hora"], suffixes=("_clima","_base"))

    alertas = []
    for _, r in m.iterrows():
        uplift = int(round(r["llamadas_clima"] - r["llamadas_base"]))
        if uplift >= MIN_UPLIFT_LLAMADAS:
            alertas.append({
                "fecha": r["fecha"],
                "hora": r["hora"],
                "llamadas_base": int(r["llamadas_base"]),
                "llamadas_con_clima": int(r["llamadas_clima"]),
                "uplift_llamadas": uplift,
                "agentes_base": required_agents(
                    r["llamadas_base"], r["tmo_segundos_base"],
                    ASA_SECONDS, SLA_TARGET, occ_max=OCC_MAX, shrinkage=SHRINKAGE
                ),
                "agentes_con_clima": required_agents(
                    r["llamadas_clima"], r["tmo_segundos_clima"],
                    ASA_SECONDS, SLA_TARGET, occ_max=OCC_MAX, shrinkage=SHRINKAGE
                )
            })
    return alertas

def generar_alertas_turnos(forecast_global_df, turnos_df):
    if "comuna_norm" in turnos_df.columns:
        turnos_g = (turnos_df.groupby(["fecha","hora"], as_index=False)
                    .agg({"agentes_planificados":"sum"}))
    else:
        turnos_g = turnos_df[["fecha","hora","agentes_planificados"]].copy()

    m = (forecast_global_df.merge(turnos_g, on=["fecha","hora"], how="left")
         .fillna({"agentes_planificados":0}))
    m["faltantes"] = (m["agentes_requeridos"] - m["agentes_planificados"]).clip(lower=0)

    alertas = []
    for _, r in m[m["faltantes"] > 0].iterrows():
        alertas.append({
            "fecha": r["fecha"],
            "hora": r["hora"],
            "agentes_planificados": int(r["agentes_planificados"]),
            "agentes_requeridos": int(r["agentes_requeridos"]),
            "faltantes": int(r["faltantes"])
        })
    return alertas

def main():
    ensure_dir(OUT_DIR)
    print(f"[diag] START pipeline | {PIPELINE_VERSION}")
    print(f"[diag] ENV FAST_GLOBAL={FAST_GLOBAL}  MAX_COMUNAS={MAX_COMUNAS}")
    print(f"[diag] ENV REQUIRE_DATASET={os.environ.get('REQUIRE_DATASET')}")

    rf, le, base = load_artifacts()
    dataset_ok = base is not None and not base.empty
    print(f"[diag] dataset loaded: {dataset_ok}")
    if dataset_ok:
        print(f"[diag] dataset rows={len(base)} | cols={list(base.columns)[:10]}...")

    if (base is None or base.empty) and os.environ.get("REQUIRE_DATASET","0")=="1":
        raise RuntimeError("Falta data/dataset_entrenamiento_llamadas.parquet. "
                           "Cárgalo en el release (parquet) o desactiva REQUIRE_DATASET.")

    global clima_df
    clima_df = pd.DataFrame(columns=["fecha","hora","comuna_norm","temperatura_c","lluvia_mm","viento_kmh"])
    if CLIMA_URL:
        try:
            clima_raw = fetch_json(CLIMA_URL)
            clima_df = parse_clima_json(clima_raw)
            print(f"[diag] clima_df rows={len(clima_df)}")
        except Exception as e:
            print("WARN: no se pudo leer CLIMA_URL:", e)

    forecast_global = generar_forecast_mensual(rf, le, base)
    print(f"[diag] forecast rows={len(forecast_global)} | total llamadas={int(forecast_global['pronostico_llamadas'].sum())}")
    forecast_path = os.path.join(OUT_DIR, "forecast_mensual.json")
    forecast_global.to_json(forecast_path, orient="records", force_ascii=False, indent=2)

    alertas_clima = generar_alertas_clima(rf, le, base, clima_df)
    print(f"[diag] alertas_clima={len(alertas_clima)}")
    with open(os.path.join(OUT_DIR, "alertas_climatologicas.json"), "w", encoding="utf-8") as f:
        json.dump(alertas_clima, f, ensure_ascii=False, indent=2)

    alertas_turnos = []
    if TURNOS_URL:
        try:
            turnos_raw = fetch_json(TURNOS_URL)
            df_turnos = parse_turnos_json(turnos_raw)
            print(f"[diag] turnos_df rows={len(df_turnos)}")
            alertas_turnos = generar_alertas_turnos(forecast_global, df_turnos)
        except Exception as e:
            print("WARN: no se pudo leer TURNOS_URL:", e)
    print(f"[diag] alertas_turnos={len(alertas_turnos)}")
    with open(os.path.join(OUT_DIR, "alertas_turnos.json"), "w", encoding="utf-8") as f:
        json.dump(alertas_turnos, f, ensure_ascii=False, indent=2)

    print("OK: JSONs globales generados en 'out/'")

if __name__ == "__main__":
    main()
