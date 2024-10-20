import cv2
import numpy as np
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
import sys
import json

def extract_texts(image_path, num_clusters,lang='en'):

    ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False, det_db_thresh=0.3, det_db_box_thresh=0.5)  # Box threshold
    
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = ocr.ocr(image_path)
    
    structured_text = structure_text(results[0] if results else [], image.shape[0], image.shape[1],num_clusters)

    plt.figure(figsize=(15, 15))
    plt.imshow(image_rgb)
    
    # Draw boxes and text
    for result in results[0] if results else []:
        boxes = np.array(result[0]).astype(np.int32).reshape((-1, 1, 2))
        
        # Draw polygon
        plt.fill(boxes[:, 0, 0], boxes[:, 0, 1], 
                alpha=0.2, color='red')
        plt.plot(boxes[:, 0, 0], boxes[:, 0, 1], 
                color='red', linewidth=2) 
        
        text = result[1][0]  # Get the text
        conf = result[1][1]  # Get the confidence score
        plt.text(boxes[0, 0, 0], boxes[0, 0, 1], f'{text} ({conf:.2f})', color='blue', fontsize=8)
    
    plt.axis('off')
    plt.savefig('detected_texts.png', bbox_inches='tight', dpi=300)
    plt.close()

    return structured_text

def structure_text(results, image_height, image_width,num_clusters):

    if not results:
        return {"No text detected": []}
    
    text_blocks = []
    positions = []  # To store the centres for clustering
    for result in results:
        boxes = np.array(result[0])
        text = result[1][0]
        confidence = result[1][1]
        
        height = np.max(boxes[:, 1]) - np.min(boxes[:, 1])
        width = np.max(boxes[:, 0]) - np.min(boxes[:, 0])
        top = np.min(boxes[:, 1])
        left = np.min(boxes[:, 0])
        area = height * width
        
        y_center = top + height / 2
        x_center = left + width / 2
        if(confidence >= 0.80):
            
            text_blocks.append({
                'text': text,
                'confidence': confidence,
                'properties': {
                    'height': height,
                    'width': width,
                    'area': area,
                    'top': top,
                    'left': left,
                    'y_center': y_center,
                    'x_center': x_center
                }
            })
            positions.append([x_center, y_center])
    
    db = KMeans(n_clusters = num_clusters).fit(positions) #The number of clusters should be given as the input

    clusters = {}
    for idx, block in enumerate(text_blocks):
        label = db.labels_[idx]
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(block)
    
    texts = {}
    for label, cluster_blocks in clusters.items():
        cluster_blocks.sort(key=lambda x: x['properties']['top'])
        header = cluster_blocks[0]['text']
        texts[header] = [block['text'] for block in cluster_blocks[1:]]

    return texts

def format(texts):
    ans_dict = {}
    for header, content in texts.items():
        if(content == []):
            ans_dict[header] = ""
            continue
        ans_dict[header] = content[0]

    return ans_dict


image_path = sys.argv[1]
num_clusters = int(sys.argv[2])

structured_text = extract_texts(image_path,num_clusters)

ans = format(structured_text)
with open('data.json', 'w') as json_file:
    json.dump(ans, json_file, indent=4)
print(ans)