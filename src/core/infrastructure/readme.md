# Esempio di utilizzo
from infrastructure.database.dependencies import (
    get_mongo_client,
    get_mongo_database,
    get_gridfs_bucket,
    get_gridfs_handler,
    get_audio_repository
)

# Ottieni le dipendenze
client = get_mongo_client()
db = get_mongo_database(client)
fs_bucket = get_gridfs_bucket(db)  # <- QUESTO
repository = get_audio_repository(db)  # <- QUESTO
gridfs_handler = get_gridfs_handler(fs_bucket)

# Oppure senza parametri (usando cache)
fs_bucket = get_gridfs_bucket()  # Automaticamente ottiene il db
repository = get_audio_repository()  # Automaticamente ottiene la collection