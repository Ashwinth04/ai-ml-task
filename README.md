# AI ML Task

This tool extracts text from images using PaddleOCR and organizes the detected text into structured clusters. It generates both a visual output showing the detected text regions and a JSON file containing the extracted text data.

## Features

- Text detection and recognition using PaddleOCR
- Clustering of text regions using K-means algorithm
- Visual output with bounding boxes and confidence scores
- JSON output of structured text data
- Support for multiple languages
- Confidence threshold filtering (>= 0.8)

## Prerequisites

Before running the script, ensure you have the following dependencies installed:

```bash
pip install opencv-python
pip install paddlepaddle
pip install "paddleocr>=2.0.1"
pip install numpy
pip install matplotlib
pip install scikit-learn
```

## Steps to run the code

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the script using the following command:

```bash
python main.py <image_path> <number_of_text_areas>
```

### Parameters:
- `image_path`: Path to the input image file
- `number_of_text_areas`: Number of expected text clusters in the image

### Example:
```bash
python main.py sample2.png 4
```

## Output

The script generates two outputs:

1. `detected_texts.png`: A visualization of the detected text regions with:
   - Red bounding boxes around detected text
   - Blue text annotations showing the detected text and confidence scores
   - Semi-transparent red highlighting of text regions

2. `data.json`: A JSON file containing the structured text data where:
   - Keys are the headers (first text in each cluster)
   - Values are the corresponding content (remaining text in the cluster)

```
{
    "Ashoka Chakra.": "Dharma Chakra (wheel of Law)",
    "Saffron": "Represents strength and courage",
    "Whited": "Symbolizes peace and truth",
    "Green": "Represents fertility, growth, and auspiciousness of the land"
}
```

## Technical Approach

### 1. Text Detection and Recognition
- Uses PaddleOCR for robust text detection and recognition
- Detection parameters:
  - `use_angle_cls=True` for rotated text detection
  - `det_db_thresh=0.3` for text detection sensitivity
  - `det_db_box_thresh=0.5` for bounding box confidence

### 2. Text Clustering
- Implements K-means clustering to group related text regions
- Clustering is based on the spatial coordinates (x_center, y_center) of text boxes
- Number of clusters is specified by the user input

### 3. Text Structuring
- Organizes detected text into hierarchical structures:
  - First text in each cluster becomes the header
  - Subsequent texts become the content
- Filters out low-confidence detections (threshold: 0.8)
- Sorts text blocks within clusters based on vertical position

### 4. Visualization
- Generates a visual representation using matplotlib
- Features include:
  - Bounding boxes for text regions
  - Confidence scores
  - Detected text display
