import psycopg2
import io
import torch
import numpy as np
from typing import List, Tuple, Optional
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_database.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FaceDatabase:
    def __init__(self, connection_params=None):
        """
        Initialize the face database with PostgreSQL + pgvector
        
        Args:
            connection_params (dict): PostgreSQL connection parameters
                Example: {
                    'dbname': 'face_recognition',
                    'user': 'postgres',
                    'password': 'password',
                    'host': 'localhost',
                    'port': '5432'
                }
        """
        self.connection_params = connection_params or {
            'dbname': 'face_recognition',
            'user': 'postgres',
            'password': 'password',
            'host': 'localhost',
            'port': '5432'
        }
        
        self.init_database()
    
    def get_connection(self):
        """Get a database connection"""
        try:
            return psycopg2.connect(**self.connection_params)
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def init_database(self):
        """Initialize database and create tables with pgvector extension"""
        try:
            # Connect to PostgreSQL
            conn = self.get_connection()
            conn.autocommit = True
            
            with conn.cursor() as cur:
                # Check if pgvector extension exists
                cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
                has_vector_ext = cur.fetchone()[0]
                
                if not has_vector_ext:
                    logger.info("Creating pgvector extension...")
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                
                # Create users table with vector type
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS face_users (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(100) UNIQUE NOT NULL,
                        embedding vector(512),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create index for similarity search
                try:
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS face_users_embedding_idx 
                        ON face_users USING ivfflat (embedding vector_cosine_ops)
                    """)
                except psycopg2.Error as e:
                    logger.warning(f"Could not create vector index: {e}")
                
                # Create recognition logs table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS recognition_logs (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES face_users(id),
                        confidence FLOAT,
                        processing_time FLOAT,
                        frame_id INTEGER,
                        client_id VARCHAR(100),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
            conn.close()
            logger.info("Database initialized successfully")
            
        except psycopg2.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    def add_user(self, username: str, embedding: torch.Tensor) -> bool:
        """
        Add new user with embedding
        
        Args:
            username (str): User's name
            embedding (torch.Tensor): Face embedding tensor
            
        Returns:
            bool: Success status
        """
        try:
            conn = self.get_connection()
            
            # Convert tensor to list for pgvector
            embedding_list = embedding.cpu().detach().numpy().flatten().tolist()
            
            with conn.cursor() as cur:
                cur.execute(
                    'INSERT INTO face_users (username, embedding) VALUES (%s, %s)',
                    (username, embedding_list)
                )
            
            conn.commit()
            conn.close()
            logger.info(f"Added user: {username}")
            return True
            
        except psycopg2.Error as e:
            logger.error(f"Error adding user: {e}")
            return False
    
    def get_all_users(self) -> Tuple[Optional[torch.Tensor], Optional[List[str]]]:
        """
        Get all users and embeddings
        
        Returns:
            Tuple[torch.Tensor, List[str]]: (embeddings, usernames)
        """
        try:
            conn = self.get_connection()
            embeddings = []
            usernames = []
            
            with conn.cursor() as cur:
                cur.execute('SELECT username, embedding FROM face_users')
                
                for username, embedding_list in cur.fetchall():
                    embedding = torch.tensor(embedding_list, dtype=torch.float32)
                    embeddings.append(embedding)
                    usernames.append(username)
            
            conn.close()
            
            if embeddings:
                embeddings_tensor = torch.stack(embeddings)
                logger.info(f"Loaded {len(usernames)} users from database")
                return embeddings_tensor, usernames
            
            logger.warning("No users found in database")
            return None, None
            
        except psycopg2.Error as e:
            logger.error(f"Error getting users: {e}")
            return None, None
    
    def find_similar_user(self, query_embedding: torch.Tensor, threshold: float = 0.6) -> Optional[Tuple[str, float]]:
        """
        Find most similar user using database vector search
        
        Args:
            query_embedding (torch.Tensor): Query face embedding
            threshold (float): Similarity threshold (0-1)
            
        Returns:
            Optional[Tuple[str, float]]: (username, similarity) or None if no match
        """
        try:
            conn = self.get_connection()
            
            # Convert tensor to list for pgvector
            query_vector = query_embedding.cpu().detach().numpy().flatten().tolist()
            
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT username, 1 - (embedding <=> %s) as similarity
                    FROM face_users
                    WHERE 1 - (embedding <=> %s) > %s
                    ORDER BY embedding <=> %s
                    LIMIT 1
                """, (query_vector, query_vector, threshold, query_vector))
                
                result = cur.fetchone()
            
            conn.close()
            
            if result:
                username, similarity = result
                logger.debug(f"Found match: {username} with similarity {similarity}")
                return username, similarity
            
            return None, 0.0
            
        except psycopg2.Error as e:
            logger.error(f"Error finding similar user: {e}")
            return None, 0.0
    
    def delete_user(self, username: str) -> bool:
        """
        Delete user by username
        
        Args:
            username (str): Username to delete
            
        Returns:
            bool: Success status
        """
        try:
            conn = self.get_connection()
            
            with conn.cursor() as cur:
                cur.execute('DELETE FROM face_users WHERE username = %s', (username,))
                deleted = cur.rowcount > 0
            
            conn.commit()
            conn.close()
            
            if deleted:
                logger.info(f"Deleted user: {username}")
                return True
            
            logger.warning(f"User not found: {username}")
            return False
            
        except psycopg2.Error as e:
            logger.error(f"Error deleting user: {e}")
            return False
    
    def log_recognition(self, user_id: int, confidence: float, processing_time: float, 
                        frame_id: int, client_id: str) -> bool:
        """
        Log a recognition event
        
        Args:
            user_id (int): User ID
            confidence (float): Recognition confidence
            processing_time (float): Processing time in seconds
            frame_id (int): Frame ID
            client_id (str): Client ID
            
        Returns:
            bool: Success status
        """
        try:
            conn = self.get_connection()
            
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO recognition_logs 
                    (user_id, confidence, processing_time, frame_id, client_id)
                    VALUES (%s, %s, %s, %s, %s)
                """, (user_id, confidence, processing_time, frame_id, client_id))
            
            conn.commit()
            conn.close()
            return True
            
        except psycopg2.Error as e:
            logger.error(f"Error logging recognition: {e}")
            return False
    
    def get_user_id(self, username: str) -> Optional[int]:
        """
        Get user ID by username
        
        Args:
            username (str): Username
            
        Returns:
            Optional[int]: User ID or None if not found
        """
        try:
            conn = self.get_connection()
            
            with conn.cursor() as cur:
                cur.execute('SELECT id FROM face_users WHERE username = %s', (username,))
                result = cur.fetchone()
            
            conn.close()
            
            if result:
                return result[0]
            
            return None
            
        except psycopg2.Error as e:
            logger.error(f"Error getting user ID: {e}")
            return None
