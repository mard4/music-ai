"""
Gestione file GridFS per storage audio.
"""
import logging
from typing import Optional, AsyncGenerator, Dict, Any
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorGridFSBucket, AsyncIOMotorGridOut

logger = logging.getLogger(__name__)


class GridFSHandler:
    """Handler per operazioni GridFS."""

    def __init__(self, fs_bucket: AsyncIOMotorGridFSBucket):
        self.fs_bucket = fs_bucket

    async def upload_file(
            self,
            file_data: bytes,
            filename: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Carica un file su GridFS.

        Args:
            file_data: Contenuto del file in bytes
            filename: Nome del file
            metadata: Metadati opzionali

        Returns:
            ID del file caricato
        """
        try:
            file_id = await self.fs_bucket.upload_from_stream(
                filename,
                file_data,
                metadata=metadata or {}
            )
            logger.info(f"File uploaded to GridFS: {filename} (ID: {file_id})")
            return str(file_id)
        except Exception as e:
            logger.error(f"Error uploading file {filename} to GridFS: {e}")
            raise

    async def download_file(self, file_id: str) -> Optional[bytes]:
        """
        Scarica un file da GridFS.

        Args:
            file_id: ID del file GridFS

        Returns:
            Contenuto del file in bytes o None se errore
        """
        try:
            grid_out = await self.fs_bucket.open_download_stream(ObjectId(file_id))
            file_data = await grid_out.read()
            logger.debug(f"Downloaded file from GridFS: {file_id}")
            return file_data
        except Exception as e:
            logger.error(f"Error downloading file {file_id} from GridFS: {e}")
            return None

    async def download_file_stream(
            self,
            file_id: str,
            chunk_size: int = 8192
    ) -> AsyncGenerator[bytes, None]:
        """
        Scarica un file in stream per gestire file grandi.

        Args:
            file_id: ID del file GridFS
            chunk_size: Dimensione dei chunk in bytes

        Yields:
            Chunk di dati del file
        """
        try:
            grid_out = await self.fs_bucket.open_download_stream(ObjectId(file_id))

            while True:
                chunk = await grid_out.readchunk()
                if not chunk:
                    break
                yield chunk

        except Exception as e:
            logger.error(f"Error streaming file {file_id} from GridFS: {e}")
            yield b''

    async def delete_file(self, file_id: str) -> bool:
        """
        Elimina un file da GridFS.

        Args:
            file_id: ID del file GridFS

        Returns:
            True se eliminato con successo
        """
        try:
            await self.fs_bucket.delete(ObjectId(file_id))
            logger.info(f"Deleted file from GridFS: {file_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting file {file_id} from GridFS: {e}")
            return False

    async def get_file_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Ottiene i metadati di un file GridFS.

        Args:
            file_id: ID del file GridFS

        Returns:
            Dizionario con metadati o None se errore
        """
        try:
            cursor = self.fs_bucket.find({"_id": ObjectId(file_id)})

            async for grid_file in cursor:
                metadata = {
                    "filename": grid_file.filename,
                    "length": grid_file.length,
                    "upload_date": grid_file.upload_date,
                    "content_type": getattr(grid_file, 'content_type', None),
                    "metadata": grid_file.metadata or {}
                }
                return metadata

            return None
        except Exception as e:
            logger.error(f"Error getting metadata for file {file_id}: {e}")
            return None

    async def file_exists(self, file_id: str) -> bool:
        """
        Verifica se un file esiste in GridFS.

        Args:
            file_id: ID del file GridFS

        Returns:
            True se il file esiste
        """
        try:
            cursor = self.fs_bucket.find({"_id": ObjectId(file_id)})
            return await cursor.fetch_next
        except Exception:
            return False

    async def list_files(
            self,
            filter_query: Optional[Dict[str, Any]] = None,
            limit: int = 100
    ) -> list:
        """
        Lista file in GridFS con filtri opzionali.

        Args:
            filter_query: Query di filtro MongoDB
            limit: Limite risultati

        Returns:
            Lista di metadati file
        """
        try:
            files = []
            cursor = self.fs_bucket.find(filter_query or {}).limit(limit)

            async for grid_file in cursor:
                file_info = {
                    "id": str(grid_file._id),
                    "filename": grid_file.filename,
                    "length": grid_file.length,
                    "upload_date": grid_file.upload_date,
                    "metadata": grid_file.metadata or {}
                }
                files.append(file_info)

            return files
        except Exception as e:
            logger.error(f"Error listing GridFS files: {e}")
            return []