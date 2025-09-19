from pydantic import BaseModel

class CustomerProfile(BaseModel):
    customer_id: str
    name: str
    email: str