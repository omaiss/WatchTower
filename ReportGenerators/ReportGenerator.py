import sqlite3
from DatabaseManagers.DatabaseManager import DatabaseManager


class ReportGenerator:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def generate_visit_frequency_report(self, days: int = 7):
        """Generate report of visit frequencies"""
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        # Get visits from last N days
        cursor.execute('''
            SELECT object_type, license_plate, COUNT(*) as visit_count,
                   MIN(timestamp) as first_visit, MAX(timestamp) as last_visit
            FROM visits 
            WHERE timestamp >= datetime('now', '-{} days')
            GROUP BY object_type, COALESCE(license_plate, 'unknown')
            ORDER BY visit_count DESC
        '''.format(days))
        
        results = cursor.fetchall()
        conn.close()
        
        print(f"\n=== Visit Frequency Report (Last {days} days) ===")
        for row in results:
            obj_type, license_plate, count, first_visit, last_visit = row
            plate_str = f" (Plate: {license_plate})" if license_plate else ""
            print(f"{obj_type.title()}{plate_str}: {count} visits")
            print(f"  First visit: {first_visit}")
            print(f"  Last visit: {last_visit}")
            print()
    
    def generate_camera_statistics(self):
        """Generate statistics by camera"""
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT c.name, v.object_type, COUNT(*) as count
            FROM cameras c
            LEFT JOIN visits v ON c.id = v.camera_id
            GROUP BY c.name, v.object_type
            ORDER BY c.name, count DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        print("\n=== Camera Statistics ===")
        current_camera = None
        for row in results:
            camera_name, object_type, count = row
            if camera_name != current_camera:
                print(f"\n{camera_name}:")
                current_camera = camera_name
            if object_type:
                print(f"  {object_type}: {count} detections")

