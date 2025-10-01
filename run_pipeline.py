name: IA Forecast + Alertas

on:
  workflow_dispatch:
  # schedule:
  #   - cron: "0 6 * * *"
  # push:
  #   branches: [ "main" ]

jobs:
  run:
    runs-on: ubuntu-latest
    permissions:
      contents: write

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'             # ✅ cache de pip para acelerar corridas siguientes

      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Install unzip & jq
        run: sudo apt-get update && sudo apt-get install -y unzip jq

      # ====== MODELOS ======
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

      # ====== DATASET PARQUET ======
      - name: Download dataset (parquet) from latest Release
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          mkdir -p data
          API="https://api.github.com/repos/${GITHUB_REPOSITORY}/releases/latest"
          DATA_URL=$(curl -s -H "Authorization: Bearer ${GH_TOKEN}" "$API" \
            | jq -r '.assets[] | select(.name=="dataset_entrenamiento_llamadas.parquet") | .url')
          if [ -z "$DATA_URL" ]; then
            echo "No se encontró dataset_entrenamiento_llamadas.parquet en el último release"; exit 1
          fi
          curl -L -H "Authorization: Bearer ${GH_TOKEN}" \
               -H "Accept: application/octet-stream" \
               "$DATA_URL" -o data/dataset_entrenamiento_llamadas.parquet
          ls -lh data

      - name: Run pipeline
        env:
          CLIMA_URL: ${{ secrets.CLIMA_URL }}
          TURNOS_URL: ${{ secrets.TURNOS_URL }}
          SLA_TARGET: ${{ secrets.SLA_TARGET }}
          ASA_SECONDS: ${{ secrets.ASA_SECONDS }}
          OCCUPANCY_MAX: ${{ secrets.OCCUPANCY_MAX }}
          SHRINKAGE: ${{ secrets.SHRINKAGE }}
          REQUIRE_DATASET: "1"
        run: |
          python run_pipeline.py

      - name: Commit JSON outputs
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add out/*.json || true
          git commit -m "chore: actualizar JSONs de forecast/alertas" || echo "Nada que commitear"
          git push
