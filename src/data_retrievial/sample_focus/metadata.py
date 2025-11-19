import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import re
from privacy_utils import get_random_user_agent

def extract_sample_metadata(url: str, session=None) -> Optional[Dict]:
    """
    Estrae i metadati da una pagina di SampleFocus.
    
    Args:
        url: L'URL della pagina SampleFocus
        scraper: Opzionale, oggetto scraper (es. cloudscraper) da usare invece di requests
        
    Returns:
        Dizionario con i metadati estratti o None in caso di errore
    """
    try:
        if session is None:
            session = requests.Session()

        headers = {
            'User-Agent': get_random_user_agent(),  
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parsing dell'HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        attrs_metadata = _extract_metadata_from_attrs(soup)

        # Estrazione dei metadati
        metadata = {
            "url": url,
            "title": _extract_title(soup),
            **attrs_metadata,
            "categories": _extract_categories(soup)
        }
        
        # Rimuovi campi None
        return {k: v for k, v in metadata.items() if v is not None}
        
    except Exception as e:
        print(f"Errore nell'estrazione dei metadati da {url}: {e}")
        return None

def _extract_title(soup: BeautifulSoup) -> Optional[str]:
    """Estrae il titolo del sample."""
    try:
        # Prova diversi selettori possibili per il titolo di SampleFocus
        title_selectors = [
            "h1[class*='title']",
            "h1[class*='sample']",
            ".sample-header h1",
            ".page-title h1",
            "h1.title",
            "h1.sample-title",
            "h1",
            # Selettori specifici per SampleFocus
            "h1.text-2xl",  # Spesso usano classi Tailwind
            "h1.text-3xl",
            "h1.font-bold",
            ".text-2xl.font-bold",  # Combinazione comune
            "[data-testid='sample-title']",  # Se usano data attributes
        ]
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                title_text = title_elem.get_text().strip()
                if title_text and len(title_text) > 1:  # Evita titoli vuoti o singoli caratteri
                    return title_text
        
        # Fallback: cerca nel meta tag og:title
        meta_title = soup.find('meta', property='og:title')
        if meta_title and meta_title.get('content'):
            return meta_title['content'].strip()
            
        # Fallback: cerca nel title della pagina
        page_title = soup.find('title')
        if page_title:
            title_text = page_title.get_text().strip()
            # Pulisci il titolo della pagina (rimuovi " - SampleFocus" etc.)
            if 'samplefocus' in title_text.lower():
                title_text = title_text.split(' - ')[0].split(' | ')[0]
            if title_text and len(title_text) > 1:
                return title_text
                
        return None
    except Exception as e:
        print(f"Errore nell'estrazione del titolo: {e}")
        return None

def _extract_metadata_from_attrs(soup: BeautifulSoup) -> Dict:
    """Estrae metadati dalla lista degli attributi (ul.sample-attrs)."""
    metadata = {}
    try:
        attrs_list = soup.find('ul', class_='sample-attrs')
        if attrs_list:
            attrs_items = attrs_list.find_all('li')
            for item in attrs_items:
                text = item.get_text().strip()
                # Controlla le icone per identificare il tipo di attributo
                if 'fa-clock' in str(item):
                    metadata['duration'] = text
                elif 'fa-ellipsis' in str(item) or 'bpm' in text.lower():
                    metadata['bpm'] = text
                elif 'fa-music' in str(item):
                    metadata['key'] = text
                # Nota: potremmo anche estrarre i favoriti se presenti
                elif 'fa-heart' in str(item):
                    try:
                        metadata['favorites_count'] = int(re.search(r'\d+', text).group())
                    except:
                        pass
    except Exception as e:
        print(f"Errore nell'estrazione degli attributi: {e}")
    return metadata

def _extract_metadata_field(soup: BeautifulSoup, field: str) -> Optional[str]:
    """Funzione helper per estrarre campi metadati comuni."""
    try:
        selectors = [
            f".sample-stats-{field}",
            f".{field}",
            f"[data-{field}]",
            f".stats-{field}"
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text().strip()
                return text if text else None
        return None
    except:
        return None

def _extract_categories(soup: BeautifulSoup) -> List[str]:
    """Estrae le categorie/tags del sample."""
    try:
        categories = []
        
        # Cerca i tag negli elementi ul.sample-tags
        tags_container = soup.select_one("ul.sample-tags")
        if tags_container:
            tag_links = tags_container.select("a.tag-link")
            categories = [tag.get_text().strip() for tag in tag_links if tag.get_text().strip()]
        
        # Se non trova niente, prova altri selettori comuni
        if not categories:
            alternative_selectors = [
                ".sample-tags a",
                ".tags a",
                ".categories a",
                "[class*='tag'] a"
            ]
            
            for selector in alternative_selectors:
                tag_elements = soup.select(selector)
                if tag_elements:
                    categories = [tag.get_text().strip() for tag in tag_elements if tag.get_text().strip()]
                    break
        
        return categories
    except:
        return []