from fastapi import FastAPI
from api.v1 import routes

app = FastAPI("Operation Engagement Platform")
app.include_router(routes.router, prefix="/api/v1")