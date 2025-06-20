from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from api.modules.routes import router

app = FastAPI()
Instrumentator().instrument(app).expose(app)

app.include_router(router, prefix="/api", tags=["API"])
