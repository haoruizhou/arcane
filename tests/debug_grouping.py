
import json
import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DebugGrouping")

SAMPLE_FOLDER = "sample_images"
CACHE_FILE = os.path.join(SAMPLE_FOLDER, ".arcane_cache.json")

def load_data():
    if not os.path.exists(CACHE_FILE):
        logger.error(f"Cache file not found at {CACHE_FILE}. Run the app and import sample_images first.")
        return []

    with open(CACHE_FILE, "r") as f:
        cache = json.load(f)
    
    data = []
    for filename, entry in cache.items():
        # Reconstruct minimal dict for grouper
        item = entry['data']
        item['path'] = os.path.join(SAMPLE_FOLDER, filename)
        # embedding is list in JSON, convert to np
        if 'embedding' in item:
            item['embedding_np'] = np.array(item['embedding'])
        
        # Ensure timestamp (mtime)
        item['timestamp'] = entry.get('mtime', 0)
        data.append(item)
        
    return data

def analyze_similarities(data):
    logger.info(f"Loaded {len(data)} items from cache.")
    
    if len(data) < 2:
        logger.info("Not enough data to analyze.")
        return

    # Sort by time
    data.sort(key=lambda x: x['timestamp'])
    
    similarities = []
    time_gaps = []
    
    print("-" * 80)
    print(f"{'Image A':<20} | {'Image B':<20} | {'Time Gap (s)':<12} | {'Cosine Sim':<10}")
    print("-" * 80)

    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            img_a = data[i]
            img_b = data[j]
            
            # Time gap
            dt = abs(img_b['timestamp'] - img_a['timestamp'])
            time_gaps.append(dt)
            
            # Cosine Sim
            emb_a = img_a.get('embedding_np')
            emb_b = img_b.get('embedding_np')
            
            if emb_a is not None and emb_b is not None:
                # Assuming normalized
                sim = np.dot(emb_a, emb_b)
                similarities.append(sim)
                
                # Print sample of high/weird ones
                if i < 3 and j < 10: # Just print a few early ones
                     print(f"{os.path.basename(img_a['path'])[:20]:<20} | {os.path.basename(img_b['path'])[:20]:<20} | {dt:<12.1f} | {sim:<10.3f}")

    if not similarities:
        logger.info("No embeddings found.")
        return

    sims = np.array(similarities)
    t_gaps = np.array(time_gaps)
    
    print("-" * 80)
    print("STATISTICS")
    print("-" * 80)
    print(f"Total Pairs: {len(sims)}")
    print(f"Similarity: Min={sims.min():.3f}, Max={sims.max():.3f}, Mean={sims.mean():.3f}, Median={np.median(sims):.3f}")
    print(f"Time Gap:   Min={t_gaps.min():.1f}s, Max={t_gaps.max():.1f}s, Mean={t_gaps.mean():.1f}s")
    
    # Check thresholds
    count_high_sim = np.sum(sims > 0.92)
    print(f"Pairs with Sim > 0.92: {count_high_sim} ({count_high_sim/len(sims)*100:.1f}%)")
    
    count_low_time = np.sum(t_gaps < 10.0)
    print(f"Pairs with Time < 10s: {count_low_time} ({count_low_time/len(t_gaps)*100:.1f}%)")
    
    # Combined (Groupable pairs)
    groupable = 0
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            dt = abs(data[i]['timestamp'] - data[j]['timestamp'])
            
            emb_a = data[i].get('embedding_np')
            emb_b = data[j].get('embedding_np')
            sim = 0
            if emb_a is not None and emb_b is not None:
                sim = np.dot(emb_a, emb_b)
                
            if dt < 10.0 and sim > 0.92:
                groupable += 1
                
    print(f"GROUPABLE Pairs (Sim>0.92 AND Time<10s): {groupable}")

if __name__ == "__main__":
    data = load_data()
    if data:
        analyze_similarities(data)
