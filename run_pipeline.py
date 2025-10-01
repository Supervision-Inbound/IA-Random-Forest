# -*- coding: utf-8 -*-
import os, re, json, math, unicodedata
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import joblib
from dateutil import parser as dtparser

# =========================
# CONFIG BÁSICA / CONSTANTES
# =========================
MODELS_DIR = "models"   # donde se descomprime models.zip (Actions lo hace)
DATA_DIR   = "data"     # aquí debe estar dataset_entrenamiento_llamadas.parquet
OUT_DIR    = "out"

# nombres preferidos de artefactos (si no coinciden, se buscarán por heurística)
PREFERRED_MODEL_NAME   = "modelo_llamadas_rf.pkl"
PREFERRED_ENCODER_NAME = "labelencoder_comunas.pkl"

# horizonte / umbrales
MESES_FORECAST       = 2         # meses hacia adelante
HORIZON_ALERTAS_H    = 24 * 7    # 1 semana (horas)
MIN_UPLIFT_LLAMADAS  = 30        # umbral de alerta climática (uplift)

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
SLA_TARGET  = get_float_env("SLA_TARGET", 0.9)     # 90%
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
    df["es_verano"]    = df["mes"].isin([12,1,2]).astype(int)
    df["es_otono"]     = df["mes"].isin([3,4,5]).astype(int)
    df["es_invierno"]  = df["mes"].isin([6,7,8]).astype(int)
    df["es_primavera"] = df["mes"].isin([9,10,11]).astype(int)
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
        print(f"[models] Encontrados (preferidos):\n  MODEL  = {preferred_model}\n  ENCODER= {preferred_encoder}")
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
        print(f"[models] Encontrados (heurística):\n  MODEL  = {guess_model}\n  ENCODER= {guess_encoder}")
        return guess_model, guess_encoder
    raise FileNotFoundError("No pude identificar modelo y encoder dentro de 'models/'.")

# =========================
# CARGA ARTEFACTOS Y DATASET
# =========================
def load_artifacts():
    model_path, encoder_path = find_artifact_paths(MODELS_DIR)
    rf = joblib.load(model_path)
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
# PREDICCIÓN ITERATIVA (por comuna; luego agregamos a global)
# =========================
FEATURES = [
    "tmo_segundos",
    "temperatura_c","lluvia_mm","viento_kmh",
    "anio","mes","dia_semana_num","es_finde",
    "es_verano","es_otono","es_invierno","es_primavera",
    "lag_1h","lag_24h","roll_mean_24h",
    "comuna_id"
]

def predict_iterativo(rf, le, base, clima_df, start_dt, horizon_hours, comunas_obj=None):
    # Fallback de histórico mínimo si no hay dataset
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

    if comunas_obj is None or len(comunas_obj)==0:
        mc = (base.groupby("comuna_norm")["conteo"].sum()
                    .sort_values(ascending=False).index[0])
        comunas_obj = [mc]

    # referencias (TMO por hora, clima de respaldo)
    tmo_col, tmo_by_hour, clima_ref = get_refs(base)
    fechas_h = pd.date_range(start=start_dt, periods=horizon_hours, freq="h")

    resultados = []
    for comuna in comunas_obj:
        dfc = base[base["comuna_norm"] == comuna].copy().sort_values("dt")
        if dfc.empty:
            dfc = base.copy()
        serie_hist = dfc[["dt","conteo"]].dropna().copy()
        for dtf in fechas_h:
            fecha = dtf.strftime("%Y-%m-%d"); hora = dtf.strftime("%H:%M")
            h_int = dtf.hour; mes_i = dtf.month
            tmo_val = tmo_by_hour.get(h_int, 210)
            rowc = pd.DataFrame()
            if not clima_df.empty:
                if "comuna_norm" in clima_df.columns and "comuna_norm" in dfc.columns:
                    rowc = clima_df[(clima_df["comuna_norm"]==comuna) & (clima_df["fecha"]==fecha) & (clima_df["hora"]==hora)]
                else:
                    rowc = clima_df[(clima_df["fecha"]==fecha) & (clima_df["hora"]==hora)]
            if len(rowc)==0:
                if "comuna_norm" in clima_ref.columns:
                    rowc = clima_ref[(clima_ref["comuna_norm"]==comuna) & (clima_ref["_mes"]==mes_i) & (clima_ref["_hora"]==h_int)]
                else:
                    rowc = clima_ref[clima_ref["_hora"]==h_int]
                if len(rowc)==0:
                    rowc = pd.DataFrame({"temperatura_c":[10],"lluvia_mm":[0.0],"viento_kmh":[15]})
            temperatura = float(rowc["temperatura_c"].iloc[0]) if "temperatura_c" in rowc else 10.0
            lluvia      = float(rowc["lluvia_mm"].iloc[0])      if "lluvia_mm" in rowc      else 0.0
            viento      = float(rowc["viento_kmh"].iloc[0])     if "viento_kmh" in rowc     else 15.0

            lag_1h  = float(serie_hist["conteo"].iloc[-1]) if len(serie_hist)>0 else 100.0
            lag_24h = float(serie_hist["conteo"].iloc[-24]) if len(serie_hist)>=24 else lag_1h
            roll_mean_24h = float(serie_hist["conteo"].tail(24).mean()) if len(serie_hist)>0 else lag_1h

            fila = pd.DataFrame({
                "fecha":[fecha], "hora":[hora],
                "tmo_segundos":[tmo_val],
                "temperatura_c":[temperatura],
                "lluvia_mm":[lluvia],
                "viento_kmh":[viento]
            })
            fila = cal_feats(fila, "fecha")

            try:
                comuna_id = int(le.transform([comuna])[0])
            except Exception:
                try:
                    comuna_id = int(le.transform([base["comuna_norm"].mode().iloc[0]])[0])
                except Exception:
                    comuna_id = 0

            fila["lag_1h"] = lag_1h
            fila["lag_24h"] = lag_24h
            fila["roll_mean_24h"] = roll_mean_24h
            fila["comuna_id"] = comuna_id

            for c in ["tmo_segundos","temperatura_c","lluvia_mm","viento_kmh","lag_1h","lag_24h","roll_mean_24h"]:
                if pd.isna(fila[c].iloc[0]):
                    fila[c] = 0.0

            yhat = float(rf.predict(fila[FEATURES])[0])
            yhat = max(0.0, yhat)
            serie_hist = pd.concat([serie_hist, pd.DataFrame({"dt":[dtf], "conteo":[yhat]})], ignore_index=True)

            resultados.append({
                "comuna": comuna,
                "fecha": fecha,
                "hora": hora,
                "llamadas": int(round(yhat)),
                "tmo_segundos": int(round(tmo_val))
            })
    return pd.DataFrame(resultados).sort_values(["comuna","fecha","hora"])

# =========================
# AGREGACIÓN A NIVEL GLOBAL
# =========================
def aggregate_global(df, y_col="llamadas", tmo_col="tmo_segundos"):
    """
    Agrega por fecha+hora a nivel global.
    - Suma llamadas
    - TMO global = promedio ponderado por llamadas (si hay llamadas; si no, mediana)
    """
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
    """
    Lista comunas a inferir para sumar global:
    - Si hay dataset con 'comuna_norm', usa esas comunas.
    - Si no, usa todas las clases del encoder; fallback: ['global'].
    """
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
    inicio = (pd.Timestamp.now() + pd.Timedelta(days=1)).normalize()
    fin    = (inicio + pd.DateOffset(months=MESES_FORECAST)).normalize()
    horas  = int((fin - inicio) / pd.Timedelta(hours=1))

    comunas_todas = list_all_comunas(le, base)

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
    out = global_df.rename(columns={"llamadas":"pronostico_llamadas", "tmo_segundos":"tmo"})
    return out[["fecha","hora","tmo","pronostico_llamadas","agentes_requeridos"]]

def generar_alertas_clima(rf, le, base, clima_df):
    start_alert = (pd.Timestamp.now() + pd.Timedelta(days=1)).normalize()
    comunas_todas = list_all_comunas(le, base)

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
    # turnos podrían venir por comuna; agregamos a global
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
    rf, le, base = load_artifacts()

    # Si exiges dataset, detén si falta
    if (base is None or base.empty) and os.environ.get("REQUIRE_DATASET","0")=="1":
        raise RuntimeError("Falta data/dataset_entrenamiento_llamadas.parquet. "
                           "Cárgalo en el release (parquet) o desactiva REQUIRE_DATASET.")

    # 1) Clima (si hay URL)
    global clima_df
    clima_df = pd.DataFrame(columns=["fecha","hora","comuna_norm","temperatura_c","lluvia_mm","viento_kmh"])
    if CLIMA_URL:
        try:
            clima_raw = fetch_json(CLIMA_URL)
            clima_df = parse_clima_json(clima_raw)
        except Exception as e:
            print("WARN: no se pudo leer CLIMA_URL:", e)

    # 2) Forecast global (2 meses)
    forecast_global = generar_forecast_mensual(rf, le, base)
    forecast_path = os.path.join(OUT_DIR, "forecast_mensual.json")
    forecast_global.to_json(forecast_path, orient="records", force_ascii=False, indent=2)

    # 3) Alertas climáticas (global)
    alertas_clima = generar_alertas_clima(rf, le, base, clima_df)
    with open(os.path.join(OUT_DIR, "alertas_climatologicas.json"), "w", encoding="utf-8") as f:
        json.dump(alertas_clima, f, ensure_ascii=False, indent=2)

    # 4) Alertas de turnos (global)
    alertas_turnos = []
    if TURNOS_URL:
        try:
            turnos_raw = fetch_json(TURNOS_URL)
            df_turnos = parse_turnos_json(turnos_raw)
            alertas_turnos = generar_alertas_turnos(forecast_global, df_turnos)
        except Exception as e:
            print("WARN: no se pudo leer TURNOS_URL:", e)
    with open(os.path.join(OUT_DIR, "alertas_turnos.json"), "w", encoding="utf-8") as f:
        json.dump(alertas_turnos, f, ensure_ascii=False, indent=2)

    print("OK: JSONs globales generados en 'out/'")
    # Resumen rápido para logs
    try:
        fg = forecast_global.copy()
        fg["y"] = pd.to_numeric(fg["pronostico_llamadas"], errors="coerce")
        tot = int(fg["y"].sum())
        dmean = round(fg.groupby(pd.to_datetime(fg["fecha"]).dt.date)["y"].sum().mean(), 2)
        print(f"[Resumen] Total periodo: {tot} | Promedio diario: {dmean}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
