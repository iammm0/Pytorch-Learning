from fastapi import FastAPI

from routes.api import router

app = FastAPI()

# 注册路由
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
