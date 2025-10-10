import psycopg2
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('db_setup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_database(db_config):
    """
    Setup PostgreSQL database and pgvector extension
    
    Args:
        db_config (dict): Database configuration
    """
    # Connect to PostgreSQL server (postgres database)
    conn_params = db_config.copy()
    conn_params['dbname'] = 'postgres'  # Connect to default postgres database first
    
    try:
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_config['dbname']}'")
        exists = cursor.fetchone()
        
        if not exists:
            logger.info(f"Creating database {db_config['dbname']}...")
            cursor.execute(f"CREATE DATABASE {db_config['dbname']}")
            logger.info(f"Database {db_config['dbname']} created successfully!")
        else:
            logger.info(f"Database {db_config['dbname']} already exists.")
        
        conn.close()
        
        # Connect to the newly created database
        conn = psycopg2.connect(**db_config)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Install pgvector extension
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            logger.info("pgvector extension installed successfully!")
        except psycopg2.Error as e:
            logger.error(f"Failed to install pgvector extension: {e}")
            logger.error("Please make sure pgvector is installed on your PostgreSQL server.")
            logger.error("Installation instructions: https://github.com/pgvector/pgvector#installation")
            return False
        
        conn.close()
        logger.info("Database setup completed successfully!")
        return True
        
    except psycopg2.Error as e:
        logger.error(f"Database setup failed: {e}")
        return False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Setup PostgreSQL database for face recognition')
    
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
    
    # Setup database
    success = setup_database(db_config)
    
    if success:
        print("\n✅ Database setup completed successfully!")
        print(f"Database: {args.db_name}")
        print(f"Host: {args.db_host}:{args.db_port}")
        print(f"User: {args.db_user}")
        print("\nYou can now run the migration script to import existing face data:")
        print(f"python migrate_to_db.py --db-name {args.db_name} --db-user {args.db_user} --db-password {args.db_password} --db-host {args.db_host} --db-port {args.db_port}")
    else:
        print("\n❌ Database setup failed. Check the logs for details.")
