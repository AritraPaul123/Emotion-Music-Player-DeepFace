import os
import json
import time
import numpy as np
import pandas as pd
import cv2
from deepface import DeepFace
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

DEEPFACE_TO_CLASS = {
    'neutral': 0,
    'happy': 1,
    'surprise': 2,
    'sad': 3,
    'angry': 4,
    'fear': 3,      # Map fear to closest category (sad/tense) or fallback
    'disgust': 4    # Map disgust to closest category (angry/aversive)
}

CLASS_NAMES = ['neutral', 'happy', 'surprise', 'sad', 'angry']

def evaluate():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'dataset')
    manifest_path = os.path.join(data_dir, 'test_manifest.csv')
    img_dir = os.path.join(data_dir, 'images')
    
    df = pd.read_csv(manifest_path)
    
    # We evaluate on a statistically significant subset of test samples (e.g., 500 samples)
    # or all samples to achieve solid comparison
    eval_df = df.sample(n=min(500, len(df)), random_state=42).reset_index(drop=True)
    print(f"Loaded test set. Evaluating DeepFace on {len(eval_df)} samples...")
    
    y_true = []
    y_pred = []
    latencies = []
    
    for idx, row in eval_df.iterrows():
        img_path = os.path.join(img_dir, row['image'])
        true_label = row['label']
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            start_t = time.perf_counter()
            res = DeepFace.analyze(
                img,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            latencies.append((time.perf_counter() - start_t) * 1000)
            
            dominant = res[0]['dominant_emotion'].lower().strip()
            pred_label = DEEPFACE_TO_CLASS.get(dominant, 0)
            
            y_true.append(true_label)
            y_pred.append(pred_label)
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(eval_df)} samples...")
        except Exception as e:
            continue
            
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    avg_latency = float(np.mean(latencies))
    
    results = {
        'model': 'DeepFace Pre-trained Emotion Model',
        'samples': len(y_true),
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'avg_latency_ms': avg_latency,
        'confusion_matrix': cm,
        'classes': CLASS_NAMES,
        'classification_report': classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    }
    
    out_path = os.path.join(base_dir, 'deepface_eval_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"DeepFace Model Accuracy: {acc*100:.2f}%")
    print(f"Weighted F1: {f1:.4f}")
    print(f"Average Inference Latency: {avg_latency:.2f} ms")
    print(f"Results saved to {out_path}")

if __name__ == '__main__':
    evaluate()

