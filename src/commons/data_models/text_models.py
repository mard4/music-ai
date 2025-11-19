"""
Modelli Pydantic per la gestione dei file i testo.
"""

from typing import Optional, List
from pydantic import BaseModel, Field

class TextFile(BaseModel):
    file_name: Optional[str] = None
    file_type: Optional[str] = None
    label: Optional[str] = None
    source: Optional[str] = None
    categories: Optional[List[str]] = None
    
class Metadata(BaseModel):
    categories: Optional[List[str]] = Field(None, description="Categorie musicali")
    

class TextFiles(BaseModel):
    text_file: TextFile
    metadata: Metadata

class EnrichedTextFile(TextFile):
    pass

class Text(BaseModel):
    file_name: str = Field(..., description="Nome del file")
    file: bytes = Field(..., description="File text in bytes")
