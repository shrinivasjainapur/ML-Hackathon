# PathSnap frontend

React, Vite, and Leaflet client for the PathSnap FastAPI service.

## Run locally

From the repository root, start the backend first:

```bash
.venv/bin/python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd frontend
npm ci
npm run dev
```

Vite serves the app at `http://127.0.0.1:5173` and proxies `/api` calls to
`http://127.0.0.1:8000`. Set `VITE_API_BASE_URL` to an absolute API URL when the
frontend and backend are hosted separately. Set `PATHSNAP_API_PROXY_TARGET`
to change the development proxy target.

The backend reads `student_test` trips from `Student folder/`, loads the model
from `outputs/models/final_model.joblib`, and optionally loads tuned HMM settings
from `outputs/models/hmm_decoder_config.json`. Paths can be changed with
`PATHSNAP_DATA_DIR`, `PATHSNAP_MODEL_PATH`, and `PATHSNAP_HMM_CONFIG_PATH`.
Check `GET /health/ready` when the frontend cannot run predictions. Inference
does not read the `.route` reference-label files.

## Checks

```bash
npm run lint
npm run build
```
