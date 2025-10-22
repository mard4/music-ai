import random
import time
from datetime import datetime

def get_random_user_agent():
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/121.0',
    ]
    return random.choice(user_agents)

class HumanBehavior:
    def __init__(self):
        self.request_count = 0
        self.session_start = datetime.now()
        self.failed_requests = 0
        
    def random_delay(self):
        """Delay casuali che sembrano umani"""
        delays = [
            random.uniform(3, 7),    # Navigazione normale
            random.uniform(8, 15),   # Lettura pagina
            random.uniform(1, 3),    # Click rapido
        ]
        delay = random.choice(delays)
        time.sleep(delay)
        return delay
    
    def browsing_pattern(self):
        """Pattern di navigazione umana"""
        if random.random() < 0.7:  # 70% del tempo
            return random.uniform(2, 5)
        else:  # 30% del tempo (più lento)
            return random.uniform(10, 20)
        
    def get_headers(self):
        """Headers che cambiano ogni volta"""
        return {
            'User-Agent': get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': random.choice(['en-US,en;q=0.5', 'it-IT,it;q=0.8,en;q=0.3', 'fr-FR,fr;q=0.9']),
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': str(random.randint(0, 1)),
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': random.choice(['max-age=0', 'no-cache']),
        }

class RateLimiter:
    def __init__(self, max_requests_per_minute=10, max_requests_per_hour=100):
        self.max_per_minute = max_requests_per_minute
        self.max_per_hour = max_requests_per_hour
        self.request_times = []

    def should_wait(self):
        """Determina se aspettare prima della prossima richiesta"""
        now = time.time()
        
        # Rimuovi richieste vecchie
        self.request_times = [t for t in self.request_times if now - t < 3600]
        
        # Controlla limite orario
        if len(self.request_times) >= self.max_per_hour:
            return True
            
        # Controlla limite minuto
        recent_requests = [t for t in self.request_times if now - t < 60]
        if len(recent_requests) >= self.max_per_minute:
            return True
            
        return False

    def record_request(self):
        """Registra una richiesta"""
        self.request_times.append(time.time())

    def get_wait_time(self):
        """Calcola quanto aspettare"""
        now = time.time()
        recent_requests = [t for t in self.request_times if now - t < 60]
        
        if len(recent_requests) >= self.max_per_minute:
            # Aspetta fino al prossimo minuto
            oldest = min(recent_requests)
            return (oldest + 60) - now
            
        return 0