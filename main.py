from pathlib import Path

import json
from html import escape

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from database import Base, engine
from environment import ActionModel, MultiTurnPrenatalEnvironment, PromptObservation, StepResult
from schemas import ResetRequest

Base.metadata.create_all(bind=engine)
BASE_DIR = Path(__file__).resolve().parent
PREVIEW_FILE = BASE_DIR / "preview.html"
MAP_DATA_FILE = BASE_DIR / "india.json"
TRAINING_GRAPH_FILE = BASE_DIR / "results" / "maas_deep_policy_demo" / "training_curve.png"
TRAINING_SUMMARY_FILE = BASE_DIR / "results" / "maas_deep_policy_demo" / "demo_summary.json"
openenv_env = MultiTurnPrenatalEnvironment()

app = FastAPI(
    title='Prenatal Health Monitor API',
    description='AI-powered prenatal health tracking system',
    version='1.0.0'
)

from routers import auth, checkin_3day, checkin_daily, coordinator, diagnosis, doctor, users

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(checkin_daily.router)
app.include_router(checkin_3day.router)
app.include_router(diagnosis.router)
app.include_router(doctor.router)
app.include_router(coordinator.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get("/", include_in_schema=False)
def root():
    if PREVIEW_FILE.exists():
        return FileResponse(PREVIEW_FILE)
    return {"message": "Prenatal Health Monitor API is running"}


@app.get("/openenv-demo", include_in_schema=False)
def openenv_demo():
    return HTMLResponse(
        """
        <html>
          <head>
            <title>MAAS OpenEnv Demo</title>
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <style>
              :root { color-scheme: light; }
              body {
                margin: 0;
                font-family: "Segoe UI", Arial, sans-serif;
                background: linear-gradient(180deg, #fff8f5 0%, #f8fafc 100%);
                color: #1f2937;
              }
              .shell {
                max-width: 1100px;
                margin: 0 auto;
                padding: 32px 20px 48px;
              }
              .hero, .card {
                background: rgba(255, 255, 255, 0.95);
                border: 1px solid #e5e7eb;
                border-radius: 24px;
                box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
              }
              .hero {
                padding: 28px;
                margin-bottom: 22px;
              }
              .hero h1 {
                margin: 0 0 10px;
                font-size: 36px;
                color: #0f172a;
              }
              .hero p {
                margin: 0;
                max-width: 760px;
                line-height: 1.6;
                color: #475569;
              }
              .links {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-top: 18px;
              }
              .links a, button {
                border: none;
                border-radius: 999px;
                padding: 10px 16px;
                font: inherit;
                font-weight: 700;
                cursor: pointer;
                text-decoration: none;
                transition: transform 0.15s ease, box-shadow 0.15s ease;
              }
              .links a, .primary {
                background: linear-gradient(135deg, #e07590, #c9a8e8);
                color: white;
                box-shadow: 0 10px 24px rgba(224, 117, 144, 0.25);
              }
              .secondary {
                background: #eef2ff;
                color: #4338ca;
              }
              .muted {
                background: #f8fafc;
                color: #334155;
              }
              .links a:hover, button:hover {
                transform: translateY(-1px);
              }
              .grid {
                display: grid;
                grid-template-columns: 1.1fr 0.9fr;
                gap: 20px;
              }
              .card {
                padding: 22px;
              }
              .card h2 {
                margin: 0 0 14px;
                font-size: 20px;
                color: #0f172a;
              }
              .stack {
                display: flex;
                flex-direction: column;
                gap: 12px;
              }
              .actions {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
              }
              textarea, pre {
                width: 100%;
                box-sizing: border-box;
                border-radius: 18px;
                border: 1px solid #dbe4f0;
                background: #0f172a;
                color: #e2e8f0;
                padding: 14px;
                font: 13px/1.5 Consolas, "Courier New", monospace;
              }
              textarea {
                min-height: 140px;
                resize: vertical;
              }
              pre {
                min-height: 220px;
                overflow: auto;
                white-space: pre-wrap;
                word-break: break-word;
              }
              .status {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                border-radius: 999px;
                padding: 8px 12px;
                background: #dcfce7;
                color: #166534;
                font-weight: 700;
              }
              .hint {
                font-size: 14px;
                color: #64748b;
                line-height: 1.6;
              }
              @media (max-width: 900px) {
                .grid {
                  grid-template-columns: 1fr;
                }
              }
            </style>
          </head>
          <body>
            <div class="shell">
              <div class="hero">
                <div class="status">Live OpenEnv deployment</div>
                <h1>MAAS Multi-Turn Demo</h1>
                <p>
                  This page lets judges run the deployed maternal-health environment end-to-end:
                  start a trajectory, inspect partial state, take OpenEnv actions, and watch
                  the episode update across turns.
                </p>
                <div class="links">
                  <a href="/docs" target="_blank" rel="noreferrer">API Docs</a>
                  <a href="/training-report" target="_blank" rel="noreferrer">Training Report</a>
                  <a href="/health" target="_blank" rel="noreferrer">Health Check</a>
                </div>
              </div>

              <div class="grid">
                <div class="card">
                  <h2>Episode Controls</h2>
                  <div class="stack">
                    <div class="actions">
                      <button class="primary" onclick="resetEpisode()">Reset Episode</button>
                      <button class="secondary" onclick="loadState()">Refresh State</button>
                      <button class="muted" onclick="applyPreset('bp')">Request BP Recheck</button>
                      <button class="muted" onclick="applyPreset('kicks')">Request Kick Count</button>
                      <button class="muted" onclick="applyPreset('advance')">Advance Day</button>
                      <button class="muted" onclick="applyPreset('phc')">Refer to PHC</button>
                      <button class="muted" onclick="applyPreset('diagnose')">Diagnose: Preeclampsia</button>
                    </div>
                    <div class="hint">
                      Use a preset action or edit the JSON manually, then submit a step.
                    </div>
                    <textarea id="actionBox">{
  "action_type": "request_bp_recheck"
}</textarea>
                    <div class="actions">
                      <button class="primary" onclick="submitStep()">Submit Step</button>
                    </div>
                  </div>
                </div>

                <div class="card">
                  <h2>Judge Flow</h2>
                  <div class="stack hint">
                    <div>1. Click <strong>Reset Episode</strong> to start a new trajectory.</div>
                    <div>2. Use one or two information-gathering actions such as BP recheck or kick count.</div>
                    <div>3. Inspect <strong>Current State</strong> to see cumulative reward and revealed observations.</div>
                    <div>4. Finish with a diagnosis action to show the final reward and rationale.</div>
                  </div>
                </div>
              </div>

              <div class="grid" style="margin-top: 20px;">
                <div class="card">
                  <h2>Last Response</h2>
                  <pre id="responseBox">Press "Reset Episode" to begin.</pre>
                </div>
                <div class="card">
                  <h2>Current State</h2>
                  <pre id="stateBox">No active trajectory yet.</pre>
                </div>
              </div>
            </div>

            <script>
              const presets = {
                bp: { action_type: "request_bp_recheck" },
                kicks: { action_type: "request_kick_count" },
                advance: { action_type: "advance_day" },
                phc: { action_type: "refer_to_phc" },
                diagnose: {
                  action_type: "diagnose",
                  condition: "preeclampsia",
                  urgency: "go_to_hospital_today",
                  rationale: "Danger signs plus elevated blood pressure warrant same-day escalation."
                }
              };

              function formatJson(payload) {
                return JSON.stringify(payload, null, 2);
              }

              function applyPreset(name) {
                document.getElementById("actionBox").value = formatJson(presets[name]);
              }

              async function loadState() {
                const response = await fetch("/state");
                const payload = await response.json();
                document.getElementById("stateBox").textContent = formatJson(payload);
                return payload;
              }

              async function resetEpisode() {
                const response = await fetch("/reset", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({})
                });
                const payload = await response.json();
                document.getElementById("responseBox").textContent = formatJson(payload);
                await loadState();
              }

              async function submitStep() {
                let action;
                try {
                  action = JSON.parse(document.getElementById("actionBox").value);
                } catch (error) {
                  document.getElementById("responseBox").textContent = "Invalid JSON in action box.";
                  return;
                }
                const response = await fetch("/step", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify(action)
                });
                const payload = await response.json();
                document.getElementById("responseBox").textContent = formatJson(payload);
                await loadState();
              }

              loadState();
            </script>
          </body>
        </html>
        """
    )

@app.get("/health", tags=["System"])
def healthcheck(request: Request):
    payload = {"status": "healthy"}
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return HTMLResponse(
            """
            <html>
              <head>
                <title>MAAS Health</title>
                <style>
                  body { font-family: Segoe UI, Arial, sans-serif; margin: 32px; background: #f8fafc; color: #0f172a; }
                  .card { max-width: 520px; background: white; border: 1px solid #cbd5e1; border-radius: 16px; padding: 24px; }
                  .pill { display: inline-block; background: #dcfce7; color: #166534; padding: 6px 12px; border-radius: 999px; font-weight: 600; }
                  p { color: #475569; }
                </style>
              </head>
              <body>
                <div class="card">
                  <h1>MAAS Health Check</h1>
                  <p class="pill">Status: healthy</p>
                  <p>The MAAS server is running and responding.</p>
                </div>
              </body>
            </html>
            """
        )
    return payload


@app.get("/health-page", include_in_schema=False)
def health_page():
    return HTMLResponse(
        """
        <html>
          <head>
            <title>MAAS Health</title>
            <style>
              body { font-family: Segoe UI, Arial, sans-serif; margin: 32px; background: #f8fafc; color: #0f172a; }
              .card { max-width: 520px; background: white; border: 1px solid #cbd5e1; border-radius: 16px; padding: 24px; }
              .pill { display: inline-block; background: #dcfce7; color: #166534; padding: 6px 12px; border-radius: 999px; font-weight: 600; }
              p { color: #475569; }
            </style>
          </head>
          <body>
            <div class="card">
              <h1>MAAS Health Check</h1>
              <p class="pill">Status: healthy</p>
              <p>The MAAS server is running and responding.</p>
            </div>
          </body>
        </html>
        """
    )


@app.get("/india-map", include_in_schema=False)
def india_map():
    if not MAP_DATA_FILE.exists():
        raise HTTPException(status_code=404, detail="India map data not found")
    return FileResponse(MAP_DATA_FILE, media_type="application/geo+json")


@app.get("/training-graph", include_in_schema=False)
def training_graph():
    if not TRAINING_GRAPH_FILE.exists():
        raise HTTPException(status_code=404, detail="Training graph not found")
    return FileResponse(TRAINING_GRAPH_FILE, media_type="image/png")


@app.get("/training-summary", include_in_schema=False)
def training_summary():
    if not TRAINING_SUMMARY_FILE.exists():
        raise HTTPException(status_code=404, detail="Training summary not found")
    return FileResponse(TRAINING_SUMMARY_FILE, media_type="application/json")


@app.get("/training-report", include_in_schema=False)
def training_report():
    if not TRAINING_SUMMARY_FILE.exists():
        raise HTTPException(status_code=404, detail="Training summary not found")
    summary = json.loads(TRAINING_SUMMARY_FILE.read_text(encoding="utf-8"))
    final_epoch = summary.get("final_epoch", {})
    metrics_json = escape(json.dumps(final_epoch, indent=2))
    return HTMLResponse(
        f"""
        <html>
          <head>
            <title>MAAS Training Report</title>
            <style>
              body {{ font-family: Segoe UI, Arial, sans-serif; margin: 32px; background: #f8fafc; color: #0f172a; }}
              .card {{ max-width: 1080px; background: white; border: 1px solid #cbd5e1; border-radius: 16px; padding: 24px; }}
              .chips {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 12px 0 20px; }}
              .chip {{ background: #e2e8f0; color: #1e293b; padding: 8px 12px; border-radius: 999px; font-weight: 600; }}
              img {{ width: 100%; max-width: 980px; border: 1px solid #cbd5e1; border-radius: 12px; }}
              pre {{ background: #0f172a; color: #e2e8f0; padding: 18px; border-radius: 12px; overflow: auto; }}
            </style>
          </head>
          <body>
            <div class="card">
              <h1>MAAS Training Report</h1>
              <div class="chips">
                <span class="chip">Val condition acc: {final_epoch.get("val_condition_acc", "n/a")}</span>
                <span class="chip">Val urgency acc: {final_epoch.get("val_urgency_acc", "n/a")}</span>
                <span class="chip">Val loss: {final_epoch.get("val_loss", "n/a")}</span>
              </div>
              <p>Training curve from the current MAAS deep-policy demo run.</p>
              <img src="/training-graph" alt="MAAS training curve" />
              <h2>Final Epoch Metrics</h2>
              <pre>{metrics_json}</pre>
            </div>
          </body>
        </html>
        """
    )

@app.post("/reset", response_model=PromptObservation, tags=["OpenEnv"])
def reset_environment(request: ResetRequest | None = None):
    payload = request or ResetRequest()
    try:
        return openenv_env.reset(trajectory_id=payload.trajectory_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/step", response_model=StepResult, tags=["OpenEnv"])
def step_environment(action: ActionModel):
    try:
        return openenv_env.step(action)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", tags=["OpenEnv"])
def environment_state():
    try:
        return openenv_env.state()
    except RuntimeError as exc:
        return {
            "status": "idle",
            "detail": str(exc),
        }
