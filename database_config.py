import psycopg2
import psycopg2.extras
from datetime import datetime
import logging
import os
from typing import Optional, Dict, Any

class DatabaseConfig:
    """Database configuration and connection management for ANPR system"""
    
    def __init__(self):
        # Database connection parameters
        self.db_config = {
            'host': '',
            'port': 25060,
            'dbname': '',
            'user': '',
            'password': '',
            'sslmode': ''
        }
        
        # Default values for required fields
        self.default_camera_id = os.getenv('ANPR_CAMERA_ID', '53b3850d-e0ef-4668-9fb5-12c980aac83d')
        self.default_location = os.getenv('ANPR_LOCATION', 'SALES OUT')
        self.default_office_id = os.getenv('ANPR_OFFICE_ID', '63a00824-f790-4518-9686-e095abcbdd8c')
        
        self.connection = None
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> bool:
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            self.connection.autocommit = True
            self.logger.info("Successfully connected to PostgreSQL database")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            self.logger.info("Database connection closed")
    
    def is_connected(self) -> bool:
        """Check if database connection is active"""
        if not self.connection:
            return False
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return True
        except:
            return False
    
    def ensure_connection(self) -> bool:
        """Ensure database connection is active, reconnect if needed"""
        if not self.is_connected():
            return self.connect()
        return True
    
    def insert_anpr_log(self, 
                       plate_number: str, 
                       plate_screenshot_path: str,
                       camera_id: Optional[str] = None,
                       location: Optional[str] = None,
                       office_id: Optional[str] = None,
                       status: str = 'detected',
                       timestamp: Optional[datetime] = None) -> bool:
        """
        Insert ANPR detection log into database
        
        Args:
            plate_number: Detected plate number
            plate_screenshot_path: Path to the screenshot/output image
            camera_id: Camera identifier (optional, uses default if not provided)
            location: Location identifier (optional, uses default if not provided)
            office_id: Office identifier UUID (optional, uses default if not provided)
            status: Status of detection ('valid', 'invalid', 'detected')
            timestamp: Custom timestamp (optional, uses current time if not provided)
        
        Returns:
            bool: True if insertion successful, False otherwise
        """
        if not self.ensure_connection():
            return False
        
        try:
            # Use defaults if values not provided
            camera_id = camera_id or self.default_camera_id
            location = location or self.default_location
            office_id = office_id or self.default_office_id
            
            # Prepare the insert statement
            insert_query = """
                INSERT INTO "anprModelLogs" 
                (time, "plateNumber", "plateScreenShotPath", "cameraId", location, office_id, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            # Use provided timestamp or current time
            log_time = timestamp or datetime.now()
            
            # Execute the insert
            cursor = self.connection.cursor()
            cursor.execute(insert_query, (
                log_time,
                plate_number,
                plate_screenshot_path,
                camera_id,
                location,
                office_id,
                status
            ))
            cursor.close()
            
            self.logger.info(f"Successfully inserted ANPR log: {plate_number} at {log_time}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to insert ANPR log: {e}")
            return False
    
    def get_recent_logs(self, limit: int = 10) -> list:
        """
        Retrieve recent ANPR logs
        
        Args:
            limit: Number of recent logs to retrieve
            
        Returns:
            list: List of recent log entries
        """
        if not self.ensure_connection():
            return []
        
        try:
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("""
                SELECT * FROM "anprModelLogs" 
                ORDER BY time DESC 
                LIMIT %s
            """, (limit,))
            
            results = cursor.fetchall()
            cursor.close()
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve recent logs: {e}")
            return []
    
    def test_connection(self) -> bool:
        """Test database connection and table access"""
        if not self.ensure_connection():
            return False
        
        try:
            cursor = self.connection.cursor()
            cursor.execute('SELECT COUNT(*) FROM "anprModelLogs"')
            count = cursor.fetchone()[0]
            cursor.close()
            self.logger.info(f"Database test successful. Current log count: {count}")
            return True
        except Exception as e:
            self.logger.error(f"Database test failed: {e}")
            return False

# Global database instance
db = DatabaseConfig()