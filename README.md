# Emergency Vehicle Siren Detection using YAMNet
Real-time emergency vehicle siren detection system using YAMNet (Yet Another Music Network) deep learning model. This project aims to detect and classify emergency vehicle sirens (ambulance, police, fire truck) in real-time using audio input.

## Features

- Real-time audio processing
- Multiple emergency vehicle siren detection
- User-friendly GUI interface
- Detailed audio analysis visualization
- Automatic logging of detections
- Configurable sensitivity settings

## Requirements
```
python >= 3.8
torch
torchaudio
numpy
sounddevice
librosa
tkinter
```

## Installation
1. Clone the repository:
```
git clone https://github.com/yourusername/emergency-siren-detection.git
cd emergency-siren-detection
```
2. Install required packages:
```
pip install -r requirements.txt
```

## Usage
Run the main application:
```
python emergency_siren_detector.py
```

The GUI will appear with the following controls:

- Start/Stop button: Toggle audio detection
- Volume level indicator
- Real-time detection results
- Debug information display

## How It Works

1. Audio Processing:

- Captures real-time audio input
- Processes audio in 3-second chunks
- Converts audio to mel-spectrogram

2. Detection System:

- Uses YAMNet pre-trained model
- Detects multiple types of emergency sirens
- Implements confidence threshold for reliable detection

3. User Interface:

- Real-time audio level monitoring
- Visual alerts for detected sirens
- Debug information display
- Detection logging

## Configuration
You can modify the following parameters in the code:
```
SAMPLE_RATE = 16000
CHUNK_SIZE = 3  # seconds
CONFIDENCE_THRESHOLD = 0.5
MIN_DETECTIONS = 2
```

## Target Classes
The system detects the following emergency vehicle sounds:

- Ambulance sirens
- Police car sirens
- Fire truck sirens
- Emergency vehicle horns
- Civil defense sirens

## Contributing

1. Fork the repository
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## Contact
Mustafa Gülhan - [Github](https://github.com/mustafagulhan)
Project Kaggle Link: [Kaggle]()
