# -- coding: utf-8 --
import os, re, json, math, unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import joblib
from dateutil import parser as dtparser

# =========================
# CONFIG BÁSICA / CONSTANTES
# =========================
MODELS_DIR = "models"    # donde se descomprime models.zip (Actions lo hace)
DATA_DIR   = "data"      # aquí debe estar dataset_entrenamiento_llamadas.parquet
OUT_DIR    = "out"

# nombres preferidos de artefactos (si no coinciden, se buscarán por heurística)
PREFERRED_MODEL_NAME   = "modelo_llamadas_rf.pkl"
PREFERRED_ENCODER_NAME = "labelencoder_comunas.pkl"

# horizonte / umbrales
MESES_FORECAST       = 2       # meses hacia adelante
HORIZON_ALERTAS_H    = 24 * 7  # 1 semana (horas)
MIN_UPLIFT_LLAMADAS  = 30      # umbral de alerta climática (uplift)

# flags de performance / modo
FAST_GLOBAL = os.environ.get("FAST_GLOBAL", "1") == "1"
MAX_COMUNAS = int(os.environ.get("MAX_COMUNAS", "20"))

# =========================
# LECTURA ROBUSTA DE ENTORNO
# =========================
def get_float_env(name: str, default: float) -> float:
    """Lee un float desde env; si está vacío o inválido, devuelve default."""
    val = os.environ.get(name)
    try:
        if val is None or str(val).strip() == "":
            return default
        return float(val)
    except Exception:
        return default

# Erlang C params
SLA_TARGET  = get_float_env("SLA_TARGET", 0.9)      # 90%
ASA_SECONDS = get_float_env("ASA_SECONDS", 20.0)   # 20s
OCC_MAX     = get_float_env("OCCUPANCY_MAX", 0.85) # 85%
SHRINKAGE   = get_float_env("SHRINKAGE", 0.3)      # 30%

# URLs (pueden venir vacías)
CLIMA_URL  = os.environ.get("CLIMA_URL", "").strip()
TURNOS_URL = os.environ.get("TURNOS_URL", "").strip()

# =========================
# UTILIDADES
# =========================
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def norm_text(s):
    if pd.isna(s): return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize('NFKD', s).encode('ascii','ignore').decode('utf-8')
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def cal_feats(df, fecha_col="fecha"):
    dt = pd.to_datetime(df[fecha_col], errors="coerce")
    df["anio"] = dt.dt.year
    df["mes"]  = dt.dt.month
    df["dia_semana_num"] = dt.dt.dayofweek
    df["es_finde"] = df["dia_semana_num"].isin([5,6]).astype(int)
    df["es_verano"]     = df["mes"].isin([12,1,2]).astype(int)
    df["es_otono"]      = df["mes"].isin([3,4,5]).astype(int)
    df["es_invierno"]   = df["mes"].isin([6,7,8]).astype(int)
    df["es_primavera"]  = df["mes"].isin([9,10,11]).astype(int)
    return df

# =========================
# ERLANG C / DOTACIÓN
# =========================
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
    """
    Calcula el número de agentes requeridos usando la fórmula de Erlang C.
    """
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

# =========================
# ARTEFACTOS (.pkl) EN models/
# =========================
def find_artifact_paths(models_root: str) -> tuple[str, str]:
    root = Path(models_root)
    if not root.exists():
        raise FileNotFoundError(f"No existe la carpeta de modelos: {models_root}")
    preferred_model = None
    preferred_encoder = None
    for p in root.rglob("*.pkl"):
        name = p.name.lower()
        if name == PREFERRED_MODEL_NAME.lower():
            preferred_model = str(p)
        if name == PREFERRED_ENCODER_NAME.lower():
            preferred_encoder = str(p)
    if preferred_model and preferred_encoder:
        print(f"[models] Preferidos:\n  MODEL   = {preferred_model}\n  ENCODER = {preferred_encoder}")
        return preferred_model, preferred_encoder
    all_pkls = list(root.rglob("*.pkl"))
    if not all_pkls:
        raise FileNotFoundError("No se encontraron archivos .pkl dentro de 'models/'.")
    all_pkls_sorted = sorted(all_pkls, key=lambda p: p.stat().st_size, reverse=True)
    guess_model = str(all_pkls_sorted[0])
    guess_encoder = None
    for p in all_pkls:
        if re.search(r"label|encoder", p.name, flags=re.I):
            guess_encoder = str(p); break
    if guess_encoder is None and len(all_pkls_sorted) > 1:
        guess_encoder = str(all_pkls_sorted[-1])
    if guess_model and guess_encoder and guess_model != guess_encoder:
        print(f"[models] Heurística:\n  MODEL   = {guess_model}\n  ENCODER = {guess_encoder}")
        return guess_model, guess_encoder
    raise FileNotFoundError("No pude identificar modelo y encoder dentro de 'models/'.")

# =========================
# CARGA ARTEFACTOS Y DATASET
# =========================
def load_artifacts():
    model_path, encoder_path = find_artifact_paths(MODELS_DIR)
    rf = joblib.load(model_path)
    # acelera RF si no vino configurado
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

# =========================
# ADAPTADORES DE JSON EXTERNOS
# =========================
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

# =========================
# REFERENCIAS SI NO HAY DATASET (fallback)
# =========================
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

# =========================
# PREDICCIÓN RÁPIDA (batch)
# =========================
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

# ----------- NUEVA VERSIÓN ROBUSTA --------------
def _merge_clima(fut, clima_df, comuna_norm, clima_ref):
    """
    Garantiza que fut tenga columnas de clima; si hay clima_df las usa,
    si faltan valores completa con clima_ref (por _mes/_hora), y finalmente
    aplica defaults.
    """
    # 1) Asegura columnas en fut
    for col in ["temperatura_c", "lluvia_mm", "viento_kmh"]:
        if col not in fut.columns:
            fut[col] = np.nan

    # 2) Si viene clima_df, normaliza y mergea por fecha/hora
    if clima_df is not None and not clima_df.empty:
        df = clima_df.copy()

        # Normaliza posibles nombres alternativos
        alt = {"Temp_C": "temperatura_c", "Precip_mm": "lluvia_mm", "Viento_kmh": "viento_kmh"}
        for k, v in alt.items():
            if k in df.columns and v not in df.columns:
                df[v] = df[k]

        # Asegura columnas mínimas
        for col in ["fecha", "hora", "temperatura_c", "lluvia_mm", "viento_kmh"]:
            if col not in df.columns:
                df[col] = np.nan

        # Normaliza hora HH:MM
        df["hora"] = df["hora"].astype(str).str.slice(0, 5)

        # Filtra por comuna si existe
        if "comuna_norm" in df.columns:
            df = df[df["comuna_norm"] == comuna_norm]

        df = df[["fecha", "hora", "temperatura_c", "lluvia_mm", "viento_kmh"]].drop_duplicates()

        # Merge principal
        fut = fut.merge(df, on=["fecha", "hora"], how="left", suffixes=("", "_new"))

        # Combina columnas nuevas si quedaron con sufijo _new
        for col in ["temperatura_c", "lluvia_mm", "viento_kmh"]:
            newcol = f"{col}_new"
            if newcol in fut.columns:
                fut[col] = fut[col].combine_first(fut[newcol])
                fut.drop(columns=[newcol], inplace=True)

    # 3) Completa faltantes con clima_ref (por _mes/_hora)
    need_ref = fut[["temperatura_c", "lluvia_mm", "viento_kmh"]].isna().any(axis=1).any()
    if need_ref and clima_ref is not None and not clima_ref.empty:
        ref = clima_ref.copy()

        # Normaliza posibles nombres alternativos
        alt = {"Temp_C": "temperatura_c", "Precip_mm": "lluvia_mm", "Viento_kmh": "viento_kmh"}
        for k, v in alt.items():
            if k in ref.columns and v not in ref.columns:
                ref[v] = ref[k]

        # Intenta construir _mes/_hora si no existen en ref
        if "_mes" not in ref.columns or "_hora" not in ref.columns:
            if "fecha" in ref.columns and "hora" in ref.columns:
                dtt = pd.to_datetime(ref["fecha"] + " " + ref["hora"], errors="coerce")
                ref["_mes"] = dtt.dt.month
                ref["_hora"] = dtt.dt.hour

        # Filtra por comuna si aplica
        if "comuna_norm" in ref.columns:
            ref = ref[ref["comuna_norm"] == comuna_norm]

        # Dejamos solo claves y valores esperados
        cols_ok = ["_mes", "_hora", "temperatura_c", "lluvia_mm", "viento_kmh"]
        ref = ref[[c for c in cols_ok if c in ref.columns]].drop_duplicates()

        if {"_mes", "_hora"}.issubset(ref.columns):
            fut = fut.merge(
                ref,
                left_on=["_mes", "_hour"],
                right_on=["_mes", "_hora"],
                how="left",
                suffixes=("", "_ref"),
            )
            for col in ["temperatura_c", "lluvia_mm", "viento_kmh"]:
                rcol = f"{col}_ref"
                if rcol in fut.columns:
                    fut[col] = fut[col].fillna(fut[rcol])
                    fut.drop(columns=[rcol], inplace=True, errors="ignore")
            fut.drop(columns=["_mes_y", "_hora_y", "_mes_ref", "_hora_ref", "_mes", "_hora"], inplace=True, errors="ignore")

    # 4) Defaults finales
    fut["temperatura_c"] = fut["temperatura_c"].fillna(10.0)
    fut["lluvia_mm"]     = fut["lluvia_mm"].fillna(0.0)
    fut["viento_kmh"]    = fut["viento_kmh"].fillna(15.0)

    return fut
# ------------------------------------------------

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

def predict_iterativo(rf, le, base, clima_df, start_dt, horizon_hours, comunas_obj=None):
    print(f"[diag] predict_iterativo: horizon_hours={horizon_hours}")
    # Base mínima si no hay dataset
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

    # Comunas
    if comunas_obj is None or len(comunas_obj) == 0:
        mc = (base.groupby("comuna_norm")["conteo"].sum()
                      .sort_values(ascending=False).index[0])
        comunas_obj = [mc]
    if FAST_GLOBAL and len(comunas_obj) > 1:
        comunas_obj = comunas_obj[:1]
    else:
        comunas_obj = comunas_obj[:MAX_COMUNAS]
    print(f"[diag] predict_iterativo: comunas={len(comunas_obj)}  muestra={comunas_obj[:5]}")

    # Refs
    tmo_col, tmo_by_hour, clima_ref = get_refs(base)

    resultados = []
    for comuna in comunas_obj:
        dfc = base[base["comuna_norm"] == comuna].copy().sort_values("dt")
        if dfc.empty:
            dfc = base.copy()

        # Futuro
        fut = _build_future_frame(start_dt, horizon_hours)
        fut["tmo_segundos"] = fut["_hour"].map(tmo_by_hour).fillna(210).astype(float)
        fut = _merge_clima(fut, clima_df, comuna, clima_ref)
        fut = cal_feats(fut, "fecha")

        # comuna_id
        try:
            comuna_id = int(le.transform([comuna])[0])
        except Exception:
            try:
                comuna_id = int(le.transform([base["comuna_norm"].mode().iloc[0]])[0])
            except Exception:
                comuna_id = 0
        fut["comuna_id"] = comuna_id

        # Lags + batch
        hist_vals = dfc["conteo"].dropna().astype(float).to_numpy()
        lags = _compute_lags_numpy(hist_vals, horizon_hours)
        lag_1h_arr  = np.array([x[0] for x in lags], dtype=float)
        lag_24h_arr = np.array([x[1] for x in lags], dtype=float)
        roll_arr    = np.array([x[2] for x in lags], dtype=float)

        # Matriz de features en el orden FEATURES
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
            "comuna_id": float(comuna_id),
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

# =========================
# AGREGACIÓN A NIVEL GLOBAL
# =========================
def aggregate_global(df, y_col="llamadas", tmo_col="tmo_segundos"):
    if df.empty:
        return pd.DataFrame(columns=["fecha","hora","tmo_segundos", y_col])

    g = (df.groupby(["fecha","hora"], as_index=False)
           .agg({y_col:"sum"}).rename(columns={y_col:"llamadas"}))

    tmp = df.copy()
    tmp["w"] = tmp[y_col].clip(lower=0)
    med_tmo = (df.groupby(["fecha","hora"])[tmo_col].median()
               .rename("tmo_mediana").reset_index())
    wsum = (tmp.groupby(["fecha","hora"])["w"].sum()
               .rename("w_sum").reset_index())
    tw = (tmp.assign(wx=lambda r: r[tmo_col]*r["w"])
               .groupby(["fecha","hora"])["wx"].sum()
               .rename("wx_sum").reset_index())
    tmo_join = (med_tmo.merge(wsum, on=["fecha","hora"], how="left")
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

def list_all_comunas(le, base):
    if base is not None and "comuna_norm" in base.columns and not base["comuna_norm"].dropna().empty:
        return sorted(base["comuna_norm"].dropna().unique().tolist())
    try:
        return sorted([str(c) for c in le.classes_])
    except Exception:
        return ["global"]

# =========================
# GENERADORES (GLOBAL)
# =========================
def generar_forecast_mensual(rf, le, base):
    """Genera el pronóstico global de llamadas y agentes para los próximos MESES_FORECAST."""
    inicio = (pd.Timestamp.now() + pd.Timedelta(days=1)).normalize()
    fin    = (inicio + pd.DateOffset(months=MESES_FORECAST)).normalize()
    horas  = int((fin - inicio) / pd.Timedelta(hours=1))

    comunas_todas = list_all_comunas(le, base)
    if FAST_GLOBAL and len(comunas_todas) > 1:
        comunas_todas = comunas_todas[:1]
    else:
        comunas_todas = comunas_todas[:MAX_COMUNAS]
    print(f"[diag] comunas para predicción: {len(comunas_todas)}  muestra={comunas_todas[:5]}")
    
    # Para el forecast mensual, usamos un clima "neutro" o de referencia.
    # Por eso pasamos un DataFrame vacío, para que la predicción use los promedios históricos.
    clima_df_vacio = pd.DataFrame()

    pred_per_comuna = predict_iterativo(
        rf, le, base,
        clima_df=clima_df_vacio,
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
    out = global_df.rename(columns={"llamadas":"pronostico_llamadas", "tmo_segundos":"tmo"})
    return out[["fecha","hora","tmo","pronostico_llamadas","agentes_requeridos"]]

def generar_alertas_clima(rf, le, base, clima_df):
    start_alert = (pd.Timestamp.now() + pd.Timedelta(days=1)).normalize()
    comunas_todas = list_all_comunas(le, base)
    if FAST_GLOBAL and len(comunas_todas) > 1:
        comunas_todas = comunas_todas[:1]
    else:
        comunas_todas = comunas_todas[:MAX_COMUNAS]

    pred_clima = predict_iterativo(rf, le, base, clima_df, start_alert, HORIZON_ALERTAS_H, comunas_obj=comunas_todas)
    df_neutro = pd.DataFrame(columns=["fecha","hora","comuna_norm","temperatura_c","lluvia_mm","viento_kmh"])
    pred_base  = predict_iterativo(rf, le, base, df_neutro, start_alert, HORIZON_ALERTAS_H, comunas_obj=comunas_todas)

    g_clima = aggregate_global(pred_clima, y_col="llamadas", tmo_col="tmo_segundos")
    g_base  = aggregate_global(pred_base,  y_col="llamadas", tmo_col="tmo_segundos")

    key = ["fecha","hora"]
    m = (g_clima.merge(g_base, on=key, suffixes=("_clima","_base")))

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

    key = ["fecha","hora"]
    m = (forecast_global_df.merge(turnos_g, on=key, how="left")
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

# =========================
# MAIN
# =========================
def main():
    ensure_dir(OUT_DIR)
    print("[diag] START pipeline")
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

    # 1) Clima (si hay URL)
    clima_df = pd.DataFrame()
    if CLIMA_URL:
        try:
            clima_raw = fetch_json(CLIMA_URL)
            clima_df = parse_clima_json(clima_raw)
            print(f"[diag] clima_df rows={len(clima_df)}")
        except Exception as e:
            print(f"WARN: no se pudo leer CLIMA_URL: {e}")

    # 2) Forecast global (2 meses)
    forecast_global = generar_forecast_mensual(rf, le, base)
    print(f"[diag] forecast rows={len(forecast_global)} | total llamadas={int(forecast_global['pronostico_llamadas'].sum())}")
    forecast_path = os.path.join(OUT_DIR, "forecast_mensual.json")
    forecast_global.to_json(forecast_path, orient="records", force_ascii=False, indent=2)

    # 3) Alertas climáticas (global)
    alertas_clima = generar_alertas_clima(rf, le, base, clima_df)
    print(f"[diag] alertas_clima={len(alertas_clima)}")
    with open(os.path.join(OUT_DIR, "alertas_climatologicas.json"), "w", encoding="utf-8") as f:
        json.dump(alertas_clima, f, ensure_ascii=False, indent=2)

    # 4) Alertas de turnos (global)
    alertas_turnos = []
    if TURNOS_URL:
        try:
            turnos_raw = fetch_json(TURNOS_URL)
            df_turnos = parse_turnos_json(turnos_raw)
            print(f"[diag] turnos_df rows={len(df_turnos)}")
            alertas_turnos = generar_alertas_turnos(forecast_global, df_turnos)
        except Exception as e:
            print(f"WARN: no se pudo leer TURNOS_URL: {e}")
    print(f"[diag] alertas_turnos={len(alertas_turnos)}")
    with open(os.path.join(OUT_DIR, "alertas_turnos.json"), "w", encoding="utf-8") as f:
        json.dump(alertas_turnos, f, ensure_ascii=False, indent=2)

    print("OK: JSONs globales generados en 'out/'")

# CORRECCIÓN CRÍTICA: Usar dos guiones bajos
if __name__ == "__main__":
    main()
