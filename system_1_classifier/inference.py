import os
import json
import numpy as np
import tensorflow as tf
#from audio_to_mel import wav_to_mel_chunks
from system_1_classifier.audio_to_mel import wav_to_mel_chunks

class EngineDiagnosticInference:
    def __init__(self):
        self.model = tf.keras.models.load_model("system_1_classifier/models/fault_classifier.keras")
        with open("system_1_classifier/models/classes.json", "r") as f:
            self.classes = json.load(f)
            
    def analyze_recording(self, wav_path):
        chunks = wav_to_mel_chunks(wav_path)
        predictions = self.model.predict(chunks[..., np.newaxis])
        
        timeline = []
        detected_faults = set()
        
        for sec, pred in enumerate(predictions):
            class_idx = int(np.argmax(pred))
            confidence = float(pred[class_idx])
            fault = self.classes[class_idx]
            
            timeline.append({"second": sec, "classification": fault, "confidence": round(confidence, 3)})
            if fault != "normal" and confidence > 0.60:
                detected_faults.add(fault)
                
        return {
            "total_duration_sec": len(chunks),
            "primary_faults": list(detected_faults),
            "timeline": timeline
        }