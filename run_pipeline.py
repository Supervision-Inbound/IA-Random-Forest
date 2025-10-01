      - name: Download models.zip from latest Release
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          mkdir -p models
          API="https://api.github.com/repos/${GITHUB_REPOSITORY}/releases/latest"
          ASSET_URL=$(curl -s -H "Authorization: Bearer ${GH_TOKEN}" "$API" \
            | jq -r '.assets[] | select(.name=="models.zip") | .url')
          if [ -z "$ASSET_URL" ]; then
            echo "No se encontró models.zip en el último release"; exit 1
          fi
          curl -L -H "Authorization: Bearer ${GH_TOKEN}" \
               -H "Accept: application/octet-stream" \
               "$ASSET_URL" -o models.zip
          unzip -o models.zip -d models
          echo "Contenido de models/"; find models -maxdepth 3 -type f -name "*.pkl" -ls

      # ⬇️ NUEVO: descargar dataset (XLSX) del release
      - name: Download dataset (xlsx) from latest Release
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          mkdir -p data
          API="https://api.github.com/repos/${GITHUB_REPOSITORY}/releases/latest"
          DATA_URL=$(curl -s -H "Authorization: Bearer ${GH_TOKEN}" "$API" \
            | jq -r '.assets[] | select(.name=="dataset_entrenamiento_llamadas.xlsx") | .url')
          if [ -z "$DATA_URL" ]; then
            echo "No se encontró dataset_entrenamiento_llamadas.xlsx en el último release"; exit 1
          fi
          curl -L -H "Authorization: Bearer ${GH_TOKEN}" \
               -H "Accept: application/octet-stream" \
               "$DATA_URL" -o data/dataset_entrenamiento_llamadas.xlsx
          ls -lh data

      # ⬇️ NUEVO: convertir XLSX → Parquet con columnas estándar
      - name: Convert dataset xlsx -> parquet
        run: |
          python - << 'PY'
          import pandas as pd, pathlib, numpy as np
          p = pathlib.Path("data/dataset_entrenamiento_llamadas.xlsx")
          if not p.exists():
              raise SystemExit("No se encontró el XLSX descargado.")
          df = pd.read_excel(p)

          # Mapear nombres posibles -> estándar
          cols = {c.lower().strip(): c for c in df.columns}
          def pick(*options):
              for o in options:
                  if o in cols: return cols[o]
              return None

          c_fecha   = pick("fecha")
          c_hora    = pick("hora","hora_intervalo","intervalo","h")
          c_llam    = pick("conteo","llamadas","llamadas recibidas","llamadas_recibidas")
          c_tmo     = pick("tmo_segundos","tmo","aht","tiempo_medio_operacion","tmo (s)")

          if not (c_fecha and c_hora and c_llam):
              raise SystemExit(f"Faltan columnas mínimas. Tengo: {list(df.columns)}")

          out = pd.DataFrame({
              "fecha": pd.to_datetime(df[c_fecha], errors="coerce").dt.strftime("%Y-%m-%d"),
              "hora":  pd.to_datetime(df[c_hora].astype(str).str.slice(0,5), format="%H:%M", errors="coerce").dt.strftime("%H:%M"),
              "conteo": pd.to_numeric(df[c_llam], errors="coerce")
          }).dropna(subset=["fecha","hora"])

          # TMO: si viene en mm:ss o similar, convertir a segundos
          def to_seconds(x):
              s = str(x)
              if ":" in s:
                  parts = s.split(":")
                  if len(parts)==2:
                      m, sec = parts
                      m = pd.to_numeric(m, errors="coerce"); sec = pd.to_numeric(sec, errors="coerce")
                      return int((m if pd.notna(m) else 0)*60 + (sec if pd.notna(sec) else 0))
              val = pd.to_numeric(s, errors="coerce")
              return int(val) if pd.notna(val) else np.nan

          if c_tmo:
              out["tmo_segundos"] = df[c_tmo].map(to_seconds)
          else:
              out["tmo_segundos"] = np.nan

          # Defaults razonables
          out["conteo"] = out["conteo"].fillna(0).astype(int)
          out["tmo_segundos"] = out["tmo_segundos"].fillna(210).astype(int)

          out = out[["fecha","hora","conteo","tmo_segundos"]]
          out.to_parquet("data/dataset_entrenamiento_llamadas.parquet", index=False)
          print("Filas convertidas:", len(out))
          PY

      - name: Run pipeline
        env:
          CLIMA_URL: ${{ secrets.CLIMA_URL }}
          TURNOS_URL: ${{ secrets.TURNOS_URL }}
          SLA_TARGET: ${{ secrets.SLA_TARGET }}
          ASA_SECONDS: ${{ secrets.ASA_SECONDS }}
          OCCUPANCY_MAX: ${{ secrets.OCCUPANCY_MAX }}
          SHRINKAGE: ${{ secrets.SHRINKAGE }}
          REQUIRE_DATASET: "1"         # ⬅️ obligatorio: falla si falta dataset
        run: |
          python run_pipeline.py

