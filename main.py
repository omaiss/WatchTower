# from WebDashboard.flask_server import main
import os
from DatabaseManagers.DatabaseManager import DatabaseManager
from VideoProcessors.VideoProcessor import VideoProcessor
from ReportGenerators.ReportGenerator import ReportGenerator
    
def main():
    """Main function to run the surveillance system"""
    # Initialize database
    db_manager = DatabaseManager()
    
    # Initialize video processor
    processor = VideoProcessor(db_manager)
    
    # Initialize report generator
    reporter = ReportGenerator(db_manager)
    
    # Test videos (you'll need to download these)
    test_videos = [
        {
            "path": "test_videos/traffic.mp4",
            "camera_id": "cam_01",
            "name": "Traffic Camera 1"
        },
        {
            "path": "test_videos/parking.mp4", 
            "camera_id": "cam_02",
            "name": "Parking Lot Camera"
        },
        {
            "path": "test_videos/entrance.mp4",
            "camera_id": "cam_03", 
            "name": "Entrance Camera"
        }
    ]
    
    print("=== Surveillance Detection System ===")
    print("Make sure you have downloaded test videos to test_videos/ folder")
    print("Recommended test videos:")
    print("1. Traffic scenes with cars and license plates")
    print("2. Parking lot footage")
    print("3. Entrance/pedestrian areas")
    print("\nPress Enter to continue or Ctrl+C to exit...")
    
    try:
        input()
        
        # Create test_videos directory
        os.makedirs("test_videos", exist_ok=True)
        os.makedirs("thumbnails", exist_ok=True)
        
        # Process each test video
        for video_config in test_videos:
            if os.path.exists(video_config["path"]):
                print(f"\nProcessing {video_config['name']}...")
                processor.process_video(
                    video_config["path"],
                    video_config["camera_id"], 
                    video_config["name"]
                )
                print(f"Completed processing {video_config['name']}")
            else:
                print(f"Video not found: {video_config['path']}")
        
        # Generate reports
        print("\nGenerating reports...")
        reporter.generate_visit_frequency_report(days=1)
        reporter.generate_camera_statistics()
        
    except KeyboardInterrupt:
        print("\nSystem stopped by user")
    
    except Exception as e:
        print(f"Error: {e}")
    

if __name__ == "__main__":
    main()
    