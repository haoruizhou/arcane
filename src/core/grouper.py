import os
import logging
import numpy as np
from typing import List, Dict, Set, Tuple
try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None

logger = logging.getLogger(__name__)

class PhotoGrouper:
    """
    Handles grouping of photos based on time and similarity (dHash/content).
    """
    
    @staticmethod
    def group_by_time(images: List[str], gap_threshold: float = 60.0) -> List[List[str]]:
        """
        Groups images into "Events" based on time gaps.
        
        Args:
            images: List of image paths.
            gap_threshold: Time gap in seconds to split events.
            
        Returns:
            List of lists, where each inner list contains paths for one event.
        """
        if not images:
            return []
            
        # Ensure images are sorted by time/name
        # We assume they are passed in order, but let's be safe if we rely on it.
        # But reading file stats is expensive if we do it for all. 
        # Let's assume the caller passes them sorted or we just sort by path for stability?
        # The GUI passes them sorted by name roughly.
        # Let's just process linearly.
        
        events = []
        current_event = []
        last_time = None
        
        # Cache times to avoid repeated syscalls if possible, or just do it.
        # For thousands of images, os.path.getmtime is fast enough.
        
        for path in images:
            try:
                t = os.path.getmtime(path)
            except OSError:
                continue # Skip missing files
                
            if last_time is None:
                current_event.append(path)
                last_time = t
                continue
                
            if abs(t - last_time) > gap_threshold:
                # New Event
                if current_event:
                    events.append(current_event)
                current_event = [path]
            else:
                current_event.append(path)
            
            last_time = t
            
        if current_event:
            events.append(current_event)
            
        return events

    @staticmethod
    def group_similar(images: List[Dict], time_threshold: float = 1800.0, content_threshold: float = 0.85) -> List[List[str]]:
        """
        Groups similar images into "Stacks" using Connected Components.
        Uses Deep Learning Embeddings (Cosine Similarity) if available, falling back to dHash.
        
        Args:
            images: List of dicts containing {'path': str, 'embedding': list/np, 'dhash': str, 'timestamp': float}.
            time_threshold: Max time difference (seconds) to consider for similarity. Default 30 mins.
            content_threshold: Min Cosine Similarity (0.0-1.0) to consider similar. Default 0.85.
                               Note: For dHash fallback, we map this to a hash distance (approx).
            
        Returns:
            List of lists, where each inner list is a stack of similar images.
        """
        if not images:
            return []

        n = len(images)
        adj = {i: [] for i in range(n)}
        
        # Ensure timestamp is present and prepare embeddings
        enriched_images = []
        for img in images:
            if 'timestamp' not in img:
                try:
                    ts = os.path.getmtime(img['path'])
                except OSError:
                    ts = 0
                img['timestamp'] = ts
            
            # Ensure embedding is numpy array if present
            if 'embedding' in img and img['embedding'] is not None:
                if isinstance(img['embedding'], list):
                    img['embedding_np'] = np.array(img['embedding'])
                else:
                    img['embedding_np'] = img['embedding']
            else:
                img['embedding_np'] = None
                
            enriched_images.append(img)
            
        # Sort by timestamp
        sorted_imgs = sorted(enumerate(enriched_images), key=lambda x: x[1]['timestamp'])
        
        # Build Adjacency Graph
        for i in range(n):
            idx_i, img_i = sorted_imgs[i]
            t_i = img_i['timestamp']
            emb_i = img_i.get('embedding_np')
            h_i = img_i.get('dhash')
            
            # Look ahead
            for j in range(i + 1, n):
                idx_j, img_j = sorted_imgs[j]
                t_j = img_j['timestamp']
                
                # Time Check
                if (t_j - t_i) > time_threshold:
                    break # Sorted
                
                # Content Check
                is_similar = False
                
                emb_j = img_j.get('embedding_np')
                if emb_i is not None and emb_j is not None:
                    # Cosine Similarity
                    # Vectors are L2 normalized in Extractor, so dot product is Cosine Sim
                    sim = np.dot(emb_i, emb_j)
                    if sim >= content_threshold:
                        is_similar = True
                elif h_i and int(h_i, 16) != 0 and img_j.get('dhash'): # Fallback to dHash
                    h_j = img_j.get('dhash')
                    dist = PhotoGrouper._hamming_distance(h_i, h_j)
                    # Map 0.85 threshold to dHash? 
                    # 0.85 is high similarity. dHash < 10 is high similarity.
                    # Let's say dHash < 12 is roughly similar.
                    if dist < 12:
                        is_similar = True
                
                if is_similar:
                    adj[idx_i].append(idx_j)
                    adj[idx_j].append(idx_i)
        
        # Find Connected Components
        visited = set()
        final_stacks = []
        
        for i in range(n):
            if i not in visited:
                # BFS to find component
                component_indices = []
                q = [i]
                visited.add(i)
                while q:
                    u = q.pop(0)
                    component_indices.append(u)
                    for v in adj[u]:
                        if v not in visited:
                            visited.add(v)
                            q.append(v)
                
                # Sort indices to match input order (or time)
                component_indices.sort() # Matches input list order
                
                stack_paths = [enriched_images[idx]['path'] for idx in component_indices]
                final_stacks.append(stack_paths)

        return final_stacks

    @staticmethod
    def _hamming_distance(h1: str, h2: str) -> int:
        try:
            val1 = int(h1, 16)
            val2 = int(h2, 16)
            return bin(val1 ^ val2).count('1')
        except ValueError:
            return 100 # Max distance if invalid

    @staticmethod
    def group_by_face(images: List[Dict], eps: float = 0.6, min_samples: int = 1) -> List[List[str]]:
        """
        Groups images by Face Identity.
        Returns a list of lists, where each inner list represents a person found in those photos.
        Note: The same image may appear in multiple groups if multiple people are in it.
        """
        if not DBSCAN:
            logger.error("scikit-learn not installed. Cannot group by face.")
            return []
            
        # Collect all face embeddings
        face_embeddings = []
        face_map = [] # (image_index, detection_index)
        
        for img_idx, img in enumerate(images):
            detections = img.get('detections', [])
            for det_idx, det in enumerate(detections):
                emb = det.get('embedding')
                if emb is not None:
                    face_embeddings.append(emb)
                    face_map.append((img_idx, det_idx))
                    
        if not face_embeddings:
            return []
            
        try:
            X = np.array(face_embeddings, dtype=np.float32)

            # Sanity check
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                logger.warning("Invalid values in face embeddings. Skipping face grouping.")
                return []
            
            # Cluster
            # metric='euclidean' on normalized vectors
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=1).fit(X)
            labels = clustering.labels_
            
            # Group by label
            clusters = {}
            for idx, label in enumerate(labels):
                if label == -1:
                    continue # Noise
                    
                if label not in clusters:
                    clusters[label] = set()
                
                img_idx = face_map[idx][0]
                img_path = images[img_idx]['path']
                clusters[label].add(img_path)
                
            # Convert to list
            return [list(paths) for paths in clusters.values()]
        
        except Exception as e:
            logger.error(f"Error during face grouping: {e}")
            return []

    @staticmethod
    def group_by_semantics(images: List[Dict], eps: float = 0.2, min_samples: int = 3) -> List[List[str]]:
        """
        Groups images by Semantic Content (using CLIP embeddings).
        """
        if not DBSCAN:
            logger.error("scikit-learn not installed. Cannot group by semantics.")
            return []

        # Collect embeddings
        embeddings = []
        indices = []
        
        for idx, img in enumerate(images):
            emb = img.get('embedding')
            if emb is not None:
                if isinstance(emb, list):
                    emb = np.array(emb)
                embeddings.append(emb)
                indices.append(idx)
                
        if not embeddings:
            return []
            
        try:
            X = np.array(embeddings, dtype=np.float32)
            
            # Sanity check
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                logger.warning("Invalid values in embeddings (NaN/Inf). Skipping grouping.")
                return []
            
            # Cluster
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=1).fit(X)
            labels = clustering.labels_
            
            clusters = {}
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                    
                if label not in clusters:
                    clusters[label] = []
                    
                img_idx = indices[i]
                clusters[label].append(images[img_idx]['path'])
                
            return list(clusters.values())
            
        except Exception as e:
            logger.error(f"Error during semantic grouping: {e}")
            return []

