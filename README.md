# Tracking Project

## Overview
The Tracking Project is designed to run continuously on a PC or server connected to the cameras' DVR. The system aims to detect, log, and store images and data to facilitate easy tracking of repeated visits by individuals and vehicles.

## Features
- **RTSP Stream Retrieval**: The system retrieves RTSP streams from connected cameras.
- **Detection and Tracking**: It detects and tracks people and vehicles, capturing license plates when visible.
- **Database Logging**: A database records each visit, including details such as person/vehicle, camera, date/time, and license plate if applicable.
- **Image/Thumbnail Storage**: Each visit is accompanied by a saved image or thumbnail for visual evidence.
- **Reporting**: The system generates reports, such as "This person has visited X times this week" and "This vehicle with license plate XXX has visited Y times."
- **Web Dashboard**: A web dashboard allows users to view and filter visit history by camera, date, license plate, and person.

## Summary
In summary, the Tracking Project provides a comprehensive solution for monitoring and logging visits, enabling users to track repeated occurrences efficiently.


## Instructions

1. **Prerequisites**:
- Python
- Cameras connected to DVR over Ethernet
- GPU based PC (Rtx 3060 ~ 4060)

2. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the Program**:
- In the terminal type:
```shell
    uvicorn WebDashboard.flask_server:app --reload
```
- This will show a link like this: http://127.0.0.1:8000/. Open it to see the Dashboard on the browser.

4. **Running the Models**:
- In the dashboard go the camera management and then add your cameras. 
- After the camera has been added and connected it will start the processing using the models.
