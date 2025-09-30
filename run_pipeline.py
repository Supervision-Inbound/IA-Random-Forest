# -*- coding: utf-8 -*-
import os, re, json, math, unicodedata
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import joblib
from dateutil import parser as dtparser

# ========= CONFIG (por variables de entorno con defaults razonables) =========
MODELS_DIR = "models"
DATA_DIR   = "data"
OUT_DIR    = "out"

MODEL_PATH   = os.path.join(MODELS_DIR, "modelo_llamadas_rf.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "labelencoder_comunas.pkl")
DATASET_PATH = os.path.join(DATA_DIR, "dataset_entrenamiento_llamadas.parquet")

CLIMA_URL   = os.environ.get("CLIMA_URL", "").strip()
TURNOS_URL  = os.environ.get("TURNOS_URL", "").strip()

# Erlang C params (puedes setear por secrets)
SLA_TARGET   = float(os.environ.get("SLA_TARGET", "0.9"))     # 90%
ASA_SECONDS  = float(os.environ.get("ASA_SECONDS", "20"))     # 20s
OCC_MAX      = float(os.environ.get("OCCUPANCY_MAX", "0.85")) # 85%
SHRINKAGE    = float(os.environ.get("SHRINKAGE", "0.3"))      # 30%

# Umbrales para alertas climáticas (ajusta a tu realidad)
MIN_UPLIFT_LLAMADAS = 30        # alerta si el clima agrega ≥30 llamadas/h
MIN_PRECIP_MM       = 5.0       # lluvia relevante
MIN_VIENTO_KMH      = 30.0      # viento relevante

# Horizonte para alertas (horas desde mañana) y forecast mensual (2 meses)
HORIZON_ALERTAS_H = 24 * 7
MESES_FORECAST    = 2

# ========= UTILIDADES =========
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

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

# ========= ERLANG C =========
def erlang_c_probability(a, n):
    """
    a: tráfico en erlangs (arrivals_per_sec * AHT_sec)
    n: número de agentes
    """
    if n <= 0: 
        return 1.0
    rho = a / n
    if rho >= 1:
        return 1.0
    # fórmula clásica
    # P(wait) = ( (a^n / n!)*(n/(n-a)) ) / ( sum_{k=0}^{n-1} (a^k / k!) + (a^n / n!)*(n/(n-a)) )
    # Usamos log para estabilidad numérica
    from math import lgamma, exp, log
    # parte superior
    log_num = n*log(a) - lgamma(n+1) + log(n/(n-a))
    # sumatorio inferior
    s = 0.0
    for k in range(n):
        s += exp(k*log(a) - lgamma(k+1))
    denom = s + exp(log_num)
    return exp(log_num) / denom

def required_agents(calls_per_hour, aht_sec, asa_target_sec, sla_target, occ_max=0.85, shrinkage=0.3, n_max=1000):
    """
    Devuelve agentes requeridos (entero) para cumplir SLA y ocupación,
    ajustado por shrinkage.
    """
    if calls_per_hour < 0: calls_per_hour = 0
    lam = calls_per_hour / 3600.0   # llegadas por segundo
    a = lam * aht_sec               # erlangs

    # buscar n mínimo que cumpla:
    # 1) occupancy = a/n <= occ_max
    # 2) service level >= sla_target, con ASA target
    # Service level aproximado con Erlang C:
    # P(espera <= T) = 1 - P(wait) * exp(-(n - a) * (T / aht))
    # (T = asa_target_sec)
    n = max(1, math.ceil(a / occ_max))  # empezar desde ocupación
    T = asa_target_sec
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

    if best_n is None:
        best_n = n  # lo que haya llegado

    # aplicar shrinkage (tiempos no productivos)
    agents_with_shrink = math.ceil(best_n / (1 - shrinkage))
    return max(1, agents_with_shrink)

# ========= CARGA ARTEFACTOS Y DATASET =========
def load_artifacts():
    rf = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    base = None
    if os.path.exists(DATASET_PATH):
        base = pd.read_parquet(DATASET_PATH)
    return rf, le, base

# ========= ADAPTADORES DE JSON (CLIMA / TURNOS) =========
def fetch_json(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def parse_clima_json(raw):
    """
    Adaptador: devuélveme dataframe con columnas:
    ['fecha','hora','comuna_norm','temperatura_c','lluvia_mm','viento_kmh']
    """
    rows = []
    for item in raw:
        # Ajusta aquí si tus claves reales son distintas
        dt_val   = item.get("datetime") or item.get("fecha_hora") or item.get("dt")
        comuna   = item.get("comuna") or item.get("location") or item.get("city")
        temp_c   = item.get("temp_c") or item.get("Temp_C")
        precip   = item.get("precip_mm") or item.get("Precip_mm") or item.get("lluvia_mm")
        viento   = item.get("viento_kmh") or item.get("wind_kmh") or item.get("Viento_kmh")

        if not dt_val or not comuna: 
            continue
        try:
            dt = dtparser.parse(dt_val)
        except Exception:
            continue
        rows.append({
            "fecha": dt.strftime("%Y-%m-%d"),
            "hora":  dt.strftime("%H:%M"),
            "comuna_norm": norm_text(comuna),
            "temperatura_c": float(temp_c) if temp_c is not None else np.nan,
            "lluvia_mm": float(precip) if precip is not None else np.nan,
            "viento_kmh": float(viento) if viento is not None else np.nan
        })
    df = pd.DataFrame(rows)
    return df

def parse_turnos_json(raw):
    """
    Adaptador: dataframe con columnas ['fecha','hora','comuna_norm','agentes_planificados']
    """
    rows = []
    for item in raw:
        fecha  = item.get("fecha")
        hora   = item.get("hora")
        comuna = item.get("comuna") or item.get("location")
        agentes= item.get("agentes_planificados") or item.get("agentes") or item.get("personas")
        if not fecha or not hora or not comuna:
            continue
        rows.append({
            "fecha": fecha,
            "hora":  hora[:5],
            "comuna_norm": norm_text(comuna),
            "agentes_planificados": int(agentes) if agentes is not None else 0
        })
    return pd.DataFrame(rows)

# ========= REFERENCIAS (TMO / CLIMA TÍPICO) A PARTIR DEL DATASET =========
def build_refs(base):
    # TMO típico por hora (mediana)
    tmo_col = "tmo_segundos" if "tmo_segundos" in base.columns else ("TMO" if "TMO" in base.columns else None)
    if tmo_col is None:
        base["tmo_segundos"] = 210
        tmo_col = "tmo_segundos"
    base["_hora_int"] = pd.to_datetime(base["hora"], format="%H:%M", errors="coerce").dt.hour
    tmo_by_hour = base.groupby("_hora_int")[tmo_col].median().to_dict()

    # Clima típico por comuna/mes/hora (medianas)
    for c in ["temperatura_c","lluvia_mm","viento_kmh"]:
        if c not in base.columns:
            base[c] = np.nan
    base["_mes"]  = pd.to_datetime(base["fecha"]).dt.month
    base["_hora"] = base["_hora_int"]
    clima_ref = (base.groupby(["comuna_norm","_mes","_hora"])
                 [["temperatura_c","lluvia_mm","viento_kmh"]]
                 .median()
                 .reset_index())

    return tmo_col, tmo_by_hour, clima_ref

# ========= PREDICCIÓN ITERATIVA (usa lags con histórico) =========
FEATURES = [
    "tmo_segundos",
    "temperatura_c","lluvia_mm","viento_kmh",
    "anio","mes","dia_semana_num","es_finde",
    "es_verano","es_otono","es_invierno","es_primavera",
    "lag_1h","lag_24h","roll_mean_24h",
    "comuna_id"
]

def predict_iterativo(rf, le, base, clima_df, start_dt, horizon_hours, comunas_obj=None):
    """
    Predice hora a hora usando lags alimentados con las predicciones previas.
    - clima_df: clima futuro (fecha/hora/comuna_norm/temperatura/lluvia/viento)
    """
    base = base.copy()
    base["dt"] = pd.to_datetime(base["fecha"] + " " + base["hora"], errors="coerce")

    # elegir comunas
    if not comunas_obj:
        mc = (base.groupby("comuna_norm")["conteo"].sum().sort_values(ascending=False).index[0])
        comunas_obj = [mc]

    # refs
    tmo_col, tmo_by_hour, clima_ref = build_refs(base)

    fechas_h = pd.date_range(start=start_dt, periods=horizon_hours, freq="H")

    resultados = []
    for comuna in comunas_obj:
        dfc = base[base["comuna_norm"] == comuna].copy().sort_values("dt")
        if dfc.empty:
            continue
        # estado para lags
        serie_hist = dfc[["dt","conteo"]].dropna().copy()

        for dtf in fechas_h:
            fecha = dtf.strftime("%Y-%m-%d")
            hora  = dtf.strftime("%H:%M")
            h_int = dtf.hour
            mes_i = dtf.month

            # TMO típico por hora
            tmo_val = tmo_by_hour.get(h_int, float(np.nan))
            if pd.isna(tmo_val):
                tmo_val = base[tmo_col].median()

            # Clima futuro: intenta usar el pronóstico entregado, si no, fallback mediana histórica
            rowc = clima_df[(clima_df["comuna_norm"]==comuna) &
                            (clima_df["fecha"]==fecha) &
                            (clima_df["hora"]==hora)]
            if len(rowc)==0:
                rowc = clima_ref[(clima_ref["comuna_norm"]==comuna) &
                                 (clima_ref["_mes"]==mes_i) &
                                 (clima_ref["_hora"]==h_int)]
                if len(rowc)==0:
                    rowc = pd.DataFrame({
                        "temperatura_c":[base["temperatura_c"].median()],
                        "lluvia_mm":[base["lluvia_mm"].median()],
                        "viento_kmh":[base["viento_kmh"].median()]
                    })
            temperatura = float(rowc["temperatura_c"].iloc[0]) if "temperatura_c" in rowc else float("nan")
            lluvia      = float(rowc["lluvia_mm"].iloc[0])      if "lluvia_mm" in rowc else float("nan")
            viento      = float(rowc["viento_kmh"].iloc[0])     if "viento_kmh" in rowc else float("nan")

            # Lags desde serie + pred previas
            lag_1h  = float(serie_hist["conteo"].iloc[-1]) if len(serie_hist)>0 else np.nan
            lag_24h = float(serie_hist["conteo"].iloc[-24]) if len(serie_hist)>=24 else np.nan
            roll_mean_24h = float(serie_hist["conteo"].tail(24).mean()) if len(serie_hist)>0 else np.nan

            fila = pd.DataFrame({
                "fecha":[fecha], "hora":[hora],
                "tmo_segundos":[tmo_val],
                "temperatura_c":[temperatura],
                "lluvia_mm":[lluvia],
                "viento_kmh":[viento]
            })
            fila = cal_feats(fila, "fecha")

            # comuna_id
            try:
                comuna_id = int(le.transform([comuna])[0])
            except Exception:
                comuna_id = int(le.transform([base["comuna_norm"].mode().iloc[0]])[0])

            fila["lag_1h"] = lag_1h
            fila["lag_24h"] = lag_24h
            fila["roll_mean_24h"] = roll_mean_24h
            fila["comuna_id"] = comuna_id

            # completar NaN básicos
            for c in ["tmo_segundos","temperatura_c","lluvia_mm","viento_kmh","lag_1h","lag_24h","roll_mean_24h"]:
                if pd.isna(fila[c].iloc[0]):
                    med = base[c].median() if c in base.columns else 0.0
                    fila[c] = med

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

    df_out = pd.DataFrame(resultados).sort_values(["comuna","fecha","hora"])
    return df_out

# ========= ALERTAS CLIMA (uplift vs baseline) =========
def generar_alertas_clima(pred_clima, pred_baseline):
    # baseline: mismo horizonte pero usando clima mediano histórico (ya lo hacemos en fallback)
    # Aquí comparamos predicciones con clima real pronosticado vs predicción con clima “típico”
    # Para construir baseline, re-llamamos predict_iterativo con clima_df vacío:
    key = ["comuna","fecha","hora"]
    m = pred_clima.merge(pred_baseline, on=key, suffixes=("_clima", "_base"))
    m["uplift"] = m["llamadas_clima"] - m["llamadas_base"]
    alertas = m[m["uplift"] >= MIN_UPLIFT_LLAMADAS].copy()
    # compactar JSON de alertas
    out = []
    for _, r in alertas.iterrows():
        out.append({
            "comuna": r["comuna"],
            "fecha": r["fecha"],
            "hora": r["hora"],
            "llamadas_baseline": int(r["llamadas_base"]),
            "llamadas_con_clima": int(r["llamadas_clima"]),
            "uplift_llamadas": int(r["uplift"]),
            # recomendación de agentes extra
            "agentes_extra_recomendados": required_agents(
                calls_per_hour=r["llamadas_clima"],
                aht_sec=r.get("tmo_segundos_clima", 210),
                asa_target_sec=ASA_SECONDS, sla_target=SLA_TARGET,
                occ_max=OCC_MAX, shrinkage=SHRINKAGE
            ) - required_agents(
                calls_per_hour=r["llamadas_base"],
                aht_sec=r.get("tmo_segundos_base", 210),
                asa_target_sec=ASA_SECONDS, sla_target=SLA_TARGET,
                occ_max=OCC_MAX, shrinkage=SHRINKAGE
            )
        })
    return out

# ========= FORECAST MENSUAL (2 meses) =========
def generar_forecast_mensual(rf, le, base, clima_ref):
    inicio = (pd.Timestamp.now() + pd.Timedelta(days=1)).normalize()
    fin    = (inicio + pd.DateOffset(months=MESES_FORECAST)).normalize()
    horas  = int((fin - inicio) / pd.Timedelta(hours=1))
    # Para meses futuros, usamos clima “típico” (clima_ref) como clima_df vacío → fallback
    pred = predict_iterativo(rf, le, base, clima_df=pd.DataFrame(columns=["fecha","hora","comuna_norm","temperatura_c","lluvia_mm","viento_kmh"]),
                             start_dt=inicio, horizon_hours=horas, comunas_obj=None)
    # Agentes requeridos por Erlang C
    pred["agentes_requeridos"] = pred.apply(
        lambda r: required_agents(r["llamadas"], r["tmo_segundos"], ASA_SECONDS, SLA_TARGET, occ_max=OCC_MAX, shrinkage=SHRINKAGE),
        axis=1
    )
    # Renombrar para salida
    pred = pred.rename(columns={"llamadas":"pronostico_llamadas", "tmo_segundos":"tmo"})
    return pred[["comuna","fecha","hora","tmo","pronostico_llamadas","agentes_requeridos"]]

# ========= ALERTAS TURNOS =========
def generar_alertas_turnos(forecast_df, turnos_df):
    key = ["comuna","fecha","hora"]
    # unificar nombres
    turnos_df = turnos_df.rename(columns={"agentes_planificados":"agentes_planificados"})
    m = forecast_df.merge(turnos_df, on=key, how="left").fillna({"agentes_planificados":0})
    m["faltantes"] = (m["agentes_requeridos"] - m["agentes_planificados"]).clip(lower=0)
    alertas = m[m["faltantes"] > 0].copy()
    out = []
    for _, r in alertas.iterrows():
        out.append({
            "comuna": r["comuna"],
            "fecha": r["fecha"],
            "hora": r["hora"],
            "agentes_planificados": int(r["agentes_planificados"]),
            "agentes_requeridos": int(r["agentes_requeridos"]),
            "faltantes": int(r["faltantes"])
        })
    return out

# ========= MAIN =========
def main():
    ensure_dir(OUT_DIR)
    rf, le, base = load_artifacts()
    if base is None:
        raise FileNotFoundError("Falta data/dataset_entrenamiento_llamadas.parquet (recomendado para bootstrapping de lags/TMO/clima).")

    # refs
    _, tmo_by_hour, clima_ref = build_refs(base)

    # --- 1) PRONÓSTICO PARA ALERTAS CLIMÁTICAS (1 semana) ---
    start_alert = (pd.Timestamp.now() + pd.Timedelta(days=1)).normalize()
    # clima real (URL)
    clima_df = pd.DataFrame(columns=["fecha","hora","comuna_norm","temperatura_c","lluvia_mm","viento_kmh"])
    if CLIMA_URL:
        try:
            clima_raw = fetch_json(CLIMA_URL)
            clima_df = parse_clima_json(clima_raw)
        except Exception as e:
            print("WARN: no se pudo leer CLIMA_URL:", e)

    # pred con clima real
    pred_clima = predict_iterativo(rf, le, base, clima_df, start_alert, HORIZON_ALERTAS_H, comunas_obj=None)
    pred_clima = pred_clima.rename(columns={"llamadas":"llamadas", "tmo_segundos":"tmo_segundos"})

    # baseline (clima típico = clima_df vacío → fallback)
    pred_base = predict_iterativo(rf, le, base, clima_df=pd.DataFrame(columns=clima_df.columns),
                                  start_dt=start_alert, horizon_hours=HORIZON_ALERTAS_H, comunas_obj=None)
    pred_base = pred_base.rename(columns={"llamadas":"llamadas", "tmo_segundos":"tmo_segundos"})

    # Unir para construir alertas (con info de TMO también)
    key = ["comuna","fecha","hora"]
    m1 = pred_clima.merge(pred_base, on=key, suffixes=("_clima","_base"))

    alertas = []
    for _, r in m1.iterrows():
        uplift = int(round(r["llamadas_clima"] - r["llamadas_base"]))
        if uplift >= MIN_UPLIFT_LLAMADAS:
            alertas.append({
                "comuna": r["comuna"],
                "fecha": r["fecha"],
                "hora": r["hora"],
                "llamadas_base": int(r["llamadas_base"]),
                "llamadas_con_clima": int(r["llamadas_clima"]),
                "uplift_llamadas": uplift,
                "agentes_recomendados_base": required_agents(
                    r["llamadas_base"], r["tmo_segundos_base"],
                    ASA_SECONDS, SLA_TARGET, occ_max=OCC_MAX, shrinkage=SHRINKAGE
                ),
                "agentes_recomendados_con_clima": required_agents(
                    r["llamadas_clima"], r["tmo_segundos_clima"],
                    ASA_SECONDS, SLA_TARGET, occ_max=OCC_MAX, shrinkage=SHRINKAGE
                )
            })

    with open(os.path.join(OUT_DIR, "alertas_climatologicas.json"), "w", encoding="utf-8") as f:
        json.dump(alertas, f, ensure_ascii=False, indent=2)

    # --- 2) FORECAST MENSUAL (2 meses) ---
    forecast_mensual = generar_forecast_mensual(rf, le, base, clima_ref)
    forecast_mensual.to_json(os.path.join(OUT_DIR, "forecast_mensual.json"), orient="records", force_ascii=False, indent=2)

    # --- 3) ALERTAS DE TURNOS ---
    alertas_turnos = []
    if TURNOS_URL:
        try:
            turnos_raw = fetch_json(TURNOS_URL)
            df_turnos = parse_turnos_json(turnos_raw)
            # Comparar vs forecast_mensual en el rango que cubran los turnos
            key = ["comuna","fecha","hora"]
            merged = forecast_mensual.merge(df_turnos, on=key, how="inner")
            if not merged.empty:
                merged["faltantes"] = (merged["agentes_requeridos"] - merged["agentes_planificados"]).clip(lower=0)
                for _, r in merged[merged["faltantes"] > 0].iterrows():
                    alertas_turnos.append({
                        "comuna": r["comuna"],
                        "fecha": r["fecha"],
                        "hora": r["hora"],
                        "agentes_planificados": int(r["agentes_planificados"]),
                        "agentes_requeridos": int(r["agentes_requeridos"]),
                        "faltantes": int(r["faltantes"])
                    })
        except Exception as e:
            print("WARN: no se pudo leer TURNOS_URL:", e)

    with open(os.path.join(OUT_DIR, "alertas_turnos.json"), "w", encoding="utf-8") as f:
        json.dump(alertas_turnos, f, ensure_ascii=False, indent=2)

    print("OK: JSONs generados en 'out/'")

if __name__ == "__main__":
    main()
