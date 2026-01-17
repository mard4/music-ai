from typing import Optional, List
from pydantic import BaseModel, Field


class TextFile(BaseModel):
    """Modello base per file di testo."""
    file_name: Optional[str] = None
    file_type: Optional[str] = None
    label: Optional[str] = None
    source: Optional[str] = None
    categories: Optional[List[str]] = None


class TextMetadata(BaseModel):
    """Metadata per file di testo."""
    categories: Optional[List[str]] = Field(None, description="Categorie")


class TextDocument(BaseModel):
    """Documento di testo completo."""
    text_file: TextFile
    metadata: TextMetadata