# application/dto/chat_response_dto

from pydantic import BaseModel

class ChatResponseDTO(BaseModel):
    message: str