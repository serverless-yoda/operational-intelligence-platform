# application/dto/chat_request_dto.py
from typing import Optional
from pydantic import BaseModel

class ChatRequestDTO(BaseModel):
    prompt: str
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 512
