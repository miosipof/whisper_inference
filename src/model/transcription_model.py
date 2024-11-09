from pydantic import BaseModel


class TranscriptionResponse(BaseModel):
    """
    Modello di risposta per la trascrizione.
    """
    text: str
