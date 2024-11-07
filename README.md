# Drowning-Detector


Introducing our drowning detection system, designed to detect and alert when someone is in danger of drowning.
Our system uses computer vision algorithms to analyze video feeds from multiple cameras around a pool or other bodies of water.

The system works by using a deep learning-based object detection model to identify a person in the water. The model is trained on thousands of images and videos of people in water environments, allowing it to accurately recognize a human body in a variety of positions and lighting conditions. Once a person is detected, the system uses motion analysis algorithms to track their movements and monitor their safety.

Overall, the drowning detection system provides an innovative and effective solution for ensuring the safety of people in and around swimming pools. By leveraging advanced computer vision and machine learning technologies, the system can quickly and accurately identify potential drowning incidents and alert caregivers, ultimately helping to save lives.

## Installation

To set up the Drowning Detection project, follow these steps:

1. **Clone the Project:**
```bash
   git clone https://github.com/randhana/Drowning-Detection-.git
```
2. **Create a Virtual Environment:**
```bash
  python -m venv env
```
3. **Install Requirements:**
```bash
   pip install -r requirements.txt
```

# Usage

To run the program, follow these steps:

1. **Create a "videos" Folder:**
   - Inside the project folder, create a new folder called "videos."

2. **Add Sample Videos:**
   - Place sample videos into the "videos" folder.

3. **Run the Program:**
   ```bash
   python DrownDetect.py --source video_file_name.mp4
5. To quit the program, press the "q" key on your keyboard.


[Here's](https://youtu.be/99GdhIozAQ8) a demonstration video of our drowning detection system in action

If you are interested in YOLO object detection, read their website:

https://pjreddie.com/darknet/yolo/
