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
    if clima_df is not None and not clima_df.empty:
        if "comuna_norm" in clima_df.columns:
            c = clima_df[clima_df["comuna_norm"] == comuna_norm]
        else:
            c = clima_df
        c = c[["fecha","hora","temperatura_c","lluvia_mm","viento_kmh"]].drop_duplicates()
        fut = fut.merge(c, on=["fecha","hora"], how="left")
    # fallback con referencia por hora/mes
    if fut[["temperatura_c","lluvia_mm","viento_kmh"]].isna().any(axis=None):
        if "comuna_norm" in clima_ref.columns:
            ref = clima_ref[clima_ref["comuna_norm"] == comuna_norm][["_mes","_hora","temperatura_c","lluvia_mm","viento_kmh"]]
        else:
            ref = clima_ref[["_mes","_hora","temperatura_c","lluvia_mm","viento_kmh"]]
        fut = fut.merge(ref, left_on=["_mes","_hour"], right_on=["_mes","_hora"], how="left", suffixes=("","_ref"))
        for col in ["temperatura_c","lluvia_mm","viento_kmh"]:
            fut[col] = fut[col].fillna(fut[f"{col}_ref"])
            fut.drop(columns=[f"{col}_ref"], inplace=True, errors="ignore")
    # valores por defecto
    fut["temperatura_c"] = fut["temperatura_c"].fillna(10.0)
    fut["lluvia_mm"]     = fut["lluvia_mm"].fillna(0.0)
    fut["viento_kmh"]    = fut["viento_kmh"].fillna(15.0)
    return fut

def _compute_lags_numpy(hist_vals, horizon_len):
    # hist_vals: np.array con la historia (>=1). Usamos el último para completar.
    last = hist_vals[-1] if hist_vals.size else 100.0
    yhat = np.empty(horizon_len, dtype=float)
    lag1 = last
    # si no hay 24 previos, usamos último
    lag24_buf = hist_vals[-24] if hist_vals.size >= 24 else last
    roll_window = list(hist_vals[-24:]) if hist_vals.size >= 1 else [last]

    for i in range(horizon_len):
        # en esta etapa yhat[i] aún no existe; solo retornamos lags “previos”
        if i == 0:
            l1 = lag1
            l24 = lag24_buf
            rmean = float(np.mean(roll_window)) if roll_window else last
        else:
            l1 = yhat[i-1]
            l24 = yhat[i-24] if i >= 24 else (yhat[i-1] if i>0 else lag1)
            # rolling 24h
            if len(roll_window) >= 24:
                rmean = float(np.mean(roll_window))
            else:
                rmean = float(np.mean(roll_window))
        yield l1, l24, rmean
        # después de predecir externamente, el caller actualizará yhat[i]
        # y aquí no modificamos roll_window; se actualiza afuera
    # nota: esta función es un generador; el caller controla la escritura en yhat

def predict_iterativo(rf, le, base, clima_df, start_dt, horizon_hours, comunas_obj=None):
    # base mínima si no hay dataset
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

    # comunas a usar
    if comunas_obj is None or len(comunas_obj) == 0:
        mc = (base.groupby("comuna_norm")["conteo"].sum()
                    .sort_values(ascending=False).index[0])
        comunas_obj = [mc]

    # Fast mode: solo una comuna (global) — acelera muchísimo
    if FAST_GLOBAL and len(comunas_obj) > 1:
        comunas_obj = comunas_obj[:1]
    else:
        comunas_obj = comunas_obj[:MAX_COMUNAS]  # limita por si acaso

    # refs
    tmo_col, tmo_by_hour, clima_ref = get_refs(base)

    resultados = []
    for comuna in comunas_obj:
        dfc = base[base["comuna_norm"] == comuna].copy().sort_values("dt")
        if dfc.empty:
            dfc = base.copy()

        # futuro (vectorizado)
        fut = _build_future_frame(start_dt, horizon_hours)
        fut["tmo_segundos"] = fut["_hour"].map(tmo_by_hour).fillna(210).astype(float)

        # clima (merge único + defaults)
        fut = _merge_clima(fut, clima_df, comuna, clima_ref)

        # calendario
        fut = cal_feats(fut, "fecha")

        # comuna_id una vez
        try:
            comuna_id = int(le.transform([comuna])[0])
        except Exception:
            try:
                comuna_id = int(le.transform([base["comuna_norm"].mode().iloc[0]])[0])
            except Exception:
                comuna_id = 0
        fut["comuna_id"] = comuna_id

        # lags con numpy + predicción en batch
        hist_vals = dfc["conteo"].dropna().astype(float).to_numpy()
        H = horizon_hours
        lag_1h_arr = np.empty(H, dtype=float)
        lag_24h_arr = np.empty(H, dtype=float)
        roll_arr    = np.empty(H, dtype=float)

        # generador de lags
        lag_gen = _compute_lags_numpy(hist_vals, H)

        # features estáticas (sin lags) para todo el horizonte
        X_static = fut[[
            "tmo_segundos","temperatura_c","lluvia_mm","viento_kmh",
            "anio","mes","dia_semana_num","es_finde",
            "es_verano","es_otono","es_invierno","es_primavera",
            "comuna_id"
        ]].astype(float).to_numpy()

        yhat = np.zeros(H, dtype=float)
        # predicción iterativa pero con arrays (sin crear dataframes fila a fila)
        for i in range(H):
            l1, l24, rmean = next(lag_gen)
            lag_1h_arr[i]  = l1
            lag_24h_arr[i] = l24
            roll_arr[i]    = rmean

            # construimos vector features para este i (concatenamos estáticos + lags)
            # orden FEATURES:
            # tmo, temp, lluvia, viento, anio, mes, dia_sem, es_finde, estaciones, lag1, lag24, roll, comuna_id
            x_row = np.concatenate([X_static[i, :], np.array([l1, l24, rmean], dtype=float)])
            # reordenar según tu lista FEATURES:
            # tu FEATURES = [tmo, temp, lluvia, viento, anio, mes, dia_sem, es_finde, es_verano, es_otono, es_invierno, es_primavera, lag_1h, lag_24h, roll_mean_24h, comuna_id]
            # X_static ya lleva los 12 primeros + comuna_id; arriba concatenamos lags; falta colocar comuna_id al final si no quedó al final
            # Como construimos X_static con comuna_id al final, x_row ya está en el orden correcto (12 primeros + comuna_id al final);
            # Intercalamos lags justo antes del comuna_id:
            # Rehacemos orden para estar 100% seguros:
            base12 = X_static[i, :-1]         # 12 primeros
            comuna_only = X_static[i, -1:]    # comuna_id
            x_row = np.concatenate([base12, np.array([l1,l24,rmean], dtype=float), comuna_only])

            # predicción single row (rápida; RF internamente vectoriza)
            yhat[i] = float(rf.predict([x_row])[0])
            if yhat[i] < 0: yhat[i] = 0.0

            # actualiza rolling buffer (para promedios) de manera barata
            # (simplemente append a un deque/lista manteniendo largo <=24)
            # pero como no lo almacenamos, basta con que compute_lags_numpy asuma yhat[i] como "último" en la siguiente vuelta.
            # Aquí no necesitamos modificar nada, lag_gen usa yhat anterior.

        # empaquetar resultados para esta comuna
        out = pd.DataFrame({
            "comuna": comuna,
            "fecha": fut["fecha"].values,
            "hora":  fut["hora"].values,
            "llamadas": np.round(yhat).astype(int),
            "tmo_segundos": np.round(fut["tmo_segundos"].values).astype(int)
        })
        resultados.append(out)

    return pd.concat(resultados, ignore_index=True).sort_values(["comuna","fecha","hora"])
    main()
