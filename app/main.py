"""PowerPulse Analytics - FastAPI backend entrypoint."""
import logging
import traceback
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .database import engine, Base
from .routers import upload, eda, forecast, anomaly, whatif, calculator, appliance, recommendations, history, dashboard, meter, campus, trend, clustering, benchmark, consumption_dna

logger = logging.getLogger("powerpulse")

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="PowerPulse Analytics API",
    description="Intelligent Energy Consumption Analytics, Forecasting and Decision-Support Platform",
    version="1.0.0",
)

ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Defense in depth: any unhandled backend exception (bad file path,
    bad data, bug in a future endpoint, etc.) still returns clean JSON
    with CORS headers attached, instead of a bare 500 with no
    Access-Control-Allow-Origin header - which is what makes a real
    backend bug look like a CORS problem in the browser console.
    The full traceback is logged server-side for real debugging."""
    logger.error("Unhandled error on %s %s:\n%s", request.method, request.url.path, traceback.format_exc())

    origin = request.headers.get("origin")
    headers = {}
    if origin in ALLOWED_ORIGINS:
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Credentials"] = "true"

    return JSONResponse(
        status_code=500,
        headers=headers,
        content={
            "detail": f"Internal server error in {request.url.path}: {exc}",
            "error_type": type(exc).__name__,
        },
    )


app.include_router(dashboard.router)
app.include_router(meter.router)
app.include_router(campus.router)
app.include_router(trend.router)
app.include_router(upload.router)
app.include_router(eda.router)
app.include_router(forecast.router)
app.include_router(anomaly.router)
app.include_router(whatif.router)
app.include_router(calculator.router)
app.include_router(appliance.router)
app.include_router(recommendations.router)
app.include_router(history.router)
app.include_router(clustering.router)
app.include_router(benchmark.router)
app.include_router(consumption_dna.router)


@app.get("/")
def root():
    return {"status": "ok", "project": "PowerPulse Analytics"}


@app.get("/api/health")
def health():
    from .ml.data_pipeline import indian_dataset_available
    return {
        "status": "healthy",
        "indian_dataset_ready": indian_dataset_available(),
    }
