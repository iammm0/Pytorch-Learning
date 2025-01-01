from fastapi import APIRouter, HTTPException
from backend.models.t5_model import load_t5_model

router = APIRouter()

# 加载模型
model = load_t5_model()


@router.post("/generate")
async def generate_mindmap(input_text: str):
    try:
        result = model.generate(input_text)
        return {"mindmap": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
