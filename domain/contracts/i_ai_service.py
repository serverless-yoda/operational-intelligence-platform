from abc import ABC, abstractmethod

class IOpenAIService(ABC):
    @abstractmethod 
    async def chat(self, prompt: str)-> str:
        pass