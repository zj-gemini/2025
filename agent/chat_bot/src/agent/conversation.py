from pydantic import BaseModel


class Message(BaseModel):
    text: str
    sender: str
