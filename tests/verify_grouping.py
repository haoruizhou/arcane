
import sys
import os
import numpy as np
import logging

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.grouper import PhotoGrouper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyGrouping")

def test_face_grouping():
    logger.info("Testing Face Grouping...")
    
    # Create dummy embeddings: 2 clusters
    # Cluster A: Person 1
    emb1 = np.array([1.0, 0.0, 0.0]) # Normalized? DBSCAN uses euclidean.
    emb2 = np.array([0.9, 0.1, 0.0])
    
    # Cluster B: Person 2
    emb3 = np.array([0.0, 1.0, 0.0])
    emb4 = np.array([0.0, 0.95, 0.1])
    
    images = [
        {'path': 'img1', 'detections': [{'embedding': emb1}]},
        {'path': 'img2', 'detections': [{'embedding': emb2}]},
        {'path': 'img3', 'detections': [{'embedding': emb3}]},
        {'path': 'img4', 'detections': [{'embedding': emb4}]},
        {'path': 'img5', 'detections': []}, # No face
        {'path': 'img6', 'detections': [{'embedding': emb1}, {'embedding': emb3}]} # Two faces
    ]
    
    # Run Grouping
    # eps=0.5. Distance between emb1 and emb2 is sqrt(0.1^2 + 0.1^2) = sqrt(0.02) = 0.14 < 0.5
    # Distance between emb1 and emb3 is sqrt(1^2 + 1^2) = 1.41 > 0.5
    groups = PhotoGrouper.group_by_face(images, eps=0.5, min_samples=1)
    
    logger.info(f"Faces Groups Found: {len(groups)}")
    for g in groups:
        logger.info(f"Group: {g}")
        
    # Expect 2 groups.
    # Group 1 (Person 1): img1, img2, img6
    # Group 2 (Person 2): img3, img4, img6
    
    # Note: dict iteration order might vary but inputs are list.
    assert len(groups) == 2
    
def test_semantic_grouping():
    logger.info("Testing Semantic Grouping...")
    
    # Cluster A: Nature
    emb1 = np.array([0.5, 0.5])
    emb2 = np.array([0.51, 0.49])
    
    # Cluster B: Party
    emb3 = np.array([0.9, 0.1])
    
    images = [
        {'path': 'img1', 'embedding': emb1},
        {'path': 'img2', 'embedding': emb2},
        {'path': 'img3', 'embedding': emb3},
    ]
    
    groups = PhotoGrouper.group_by_semantics(images, eps=0.1, min_samples=1)
    
    logger.info(f"Semantic Groups Found: {len(groups)}")
    for g in groups:
        logger.info(f"Group: {g}")
        
    assert len(groups) == 2
    # Group 1: img1, img2
    # Group 2: img3

if __name__ == "__main__":
    try:
        test_face_grouping()
        test_semantic_grouping()
        print("VERIFICATION SUCCESS")
    except Exception as e:
        logger.error(f"VERIFICATION FAILED: {e}")
        sys.exit(1)
