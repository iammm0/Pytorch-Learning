from pydantic import BaseModel

class GenerateInput(BaseModel):
    input_text: str
