import torch
import numpy as np
import os
import logging
import argparse
from face_database import FaceDatabase

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def migrate_from_files_to_db(data_path='./data', db_config=None, device=None):
    """
    Migrate existing file-based face data to PostgreSQL database
    
    Args:
        data_path (str): Path to data directory
        db_config (dict): Database configuration
        device (torch.device): Device to load tensors to
    """
    logger.info("Starting migration from files to PostgreSQL database")
    
    # Initialize database
    db = FaceDatabase(connection_params=db_config)
    
    # Determine device
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load existing data
    try:
        # Determine embedding file based on device
        if device.type == 'cpu':
            embedding_file = os.path.join(data_path, "faceslistCPU.pth")
        else:
            embedding_file = os.path.join(data_path, "faceslist.pth")
        
        username_file = os.path.join(data_path, "usernames.npy")
        
        # Check if files exist
        if not os.path.exists(embedding_file):
            logger.error(f"Embedding file not found: {embedding_file}")
            return False
        
        if not os.path.exists(username_file):
            logger.error(f"Username file not found: {username_file}")
            return False
        
        # Load embeddings
        logger.info(f"Loading embeddings from {embedding_file}")
        embeddings = torch.load(embedding_file, map_location=device)
        
        # Load usernames
        logger.info(f"Loading usernames from {username_file}")
        usernames = np.load(username_file, allow_pickle=True)
        
        # Convert to list if it's a numpy array
        if isinstance(usernames, np.ndarray):
            usernames = usernames.tolist()
        
        # Ensure all usernames are strings
        usernames = [str(name) for name in usernames]
        
        logger.info(f"Found {len(usernames)} users to migrate")
        
        # Add to database
        success_count = 0
        for i, username in enumerate(usernames):
            embedding = embeddings[i:i+1]  # Single embedding
            success = db.add_user(username, embedding)
            
            if success:
                logger.info(f"✅ Migrated user: {username}")
                success_count += 1
            else:
                logger.error(f"❌ Failed to migrate user: {username}")
        
        logger.info(f"Migration completed! {success_count}/{len(usernames)} users migrated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Migrate face data from files to PostgreSQL')
    
    parser.add_argument('--data-path', type=str, default='./data',
                        help='Path to data directory')
    
    parser.add_argument('--db-name', type=str, default='face_recognition',
                        help='PostgreSQL database name')
    
    parser.add_argument('--db-user', type=str, default='postgres',
                        help='PostgreSQL username')
    
    parser.add_argument('--db-password', type=str, default='password',
                        help='PostgreSQL password')
    
    parser.add_argument('--db-host', type=str, default='localhost',
                        help='PostgreSQL host')
    
    parser.add_argument('--db-port', type=str, default='5432',
                        help='PostgreSQL port')
    
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Configure database connection
    db_config = {
        'dbname': args.db_name,
        'user': args.db_user,
        'password': args.db_password,
        'host': args.db_host,
        'port': args.db_port
    }
    
    # Set device
    device = torch.device('cpu') if args.cpu else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Run migration
    success = migrate_from_files_to_db(args.data_path, db_config, device)
    
    if success:
        print("\n✅ Migration completed successfully!")
    else:
        print("\n❌ Migration failed. Check the logs for details.")
