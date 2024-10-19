from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import FRONTEND_URL
from routes import video, utils
from dotenv import load_dotenv
import os

load_dotenv()  # .env ファイルから環境変数を読み込む

app = FastAPI()

# CORSミドルウェアを追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ルーターを追加
app.include_router(video.router)
app.include_router(utils.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
