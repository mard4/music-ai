from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class SearchResult(BaseModel):
    """Modello generico per un risultato di ricerca vettoriale."""
    score: float
    payload: Dict[str, Any]
    id: Optional[Any] = None

class AudioSearchResult(SearchResult):
    """Risultato specifico per audio."""
    filename: str
    label: str
    categories: List[str]
    metadata: Optional[Dict[str, Any]] = None

class SocialFXSearchResult(SearchResult):
    """Risultato specifico per parametri DSP."""
    descriptor: str
    effect_type: str
    param_values: List[float]
    param_keys: List[str]