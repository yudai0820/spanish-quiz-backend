import azure.functions as func
from app.main import app

# FastAPI を Azure Functions の ASGI アプリとして登録
app = func.AsgiFunctionApp(app=app, http_auth_level=func.AuthLevel.ANONYMOUS)
