
import os
import sys
import numpy as np
import cv2
import logging

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.focus_detector import FocusDetector
from src.ml.eye_detector import EyeOpennessDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyScores")

def test_focus_score():
    logger.info("Testing Focus Score Standardizaton...")
    
    # 1. Flat image (Blurry)
    flat_img = np.zeros((100, 100, 3), dtype=np.uint8)
    score_flat = FocusDetector.measure_sharpness(flat_img)
    logger.info(f"Flat Image Score (Expect ~0): {score_flat}")
    
    # 2. Noise image (Sharp)
    noise_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    score_noise = FocusDetector.measure_sharpness(noise_img)
    logger.info(f"Noise Image Score (Expect ~60-80 with K=1000): {score_noise}")
    
    # 3. Edge image (Medium)
    edge_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(edge_img, (20, 20), (80, 80), (255, 255, 255), -1)
    score_edge = FocusDetector.measure_sharpness(edge_img)
    logger.info(f"Edge Image Score (Expect ~Mid): {score_edge}")

    assert 0 <= score_flat <= 100, "Score out of range"
    assert 0 <= score_noise <= 100, "Score out of range"
    assert score_noise > score_flat, "Noise should be sharper than flat"

def test_eye_score():
    logger.info("Testing Eye Openness Score Standardizaton...")
    
    # Create a dummy eye patch
    # Flat (Closedish/No detail)
    flat_img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Landmarks: [LeftEye, RightEye, ...]
    landmarks = np.array([
        [30, 30], # Left Eye
        [70, 30], # Right Eye
        [50, 50], # Nose
        [40, 70], # Left Mouth
        [60, 70]  # Right Mouth
    ])
    
    score_flat = EyeOpennessDetector.check_eyes(flat_img, landmarks)
    logger.info(f"Flat Eye Score (Expect ~0): {score_flat}")
    
    # High Variance Eye (Open)
    # Simulate high variance around the eye point
    noise_img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Add noise around (30,30) and (70,30)
    img_h, img_w = 100, 100
    for y in range(20, 40):
        for x in range(20, 40):
            noise_img[y, x] = np.random.randint(0, 255, 3)
    for y in range(20, 40):
        for x in range(60, 80):
             noise_img[y, x] = np.random.randint(0, 255, 3)
             
    score_open = EyeOpennessDetector.check_eyes(noise_img, landmarks)
    logger.info(f"Noise Eye Score (Expect High): {score_open}")
    
    assert 0 <= score_flat <= 100, "Score out of range"
    assert 0 <= score_open <= 100, "Score out of range"
    assert score_open > score_flat, "Noise should be more open than flat"

if __name__ == "__main__":
    test_focus_score()
    test_eye_score()
