import unittest
import time
from src.core.grouper import PhotoGrouper

class TestPhotoGrouper(unittest.TestCase):
    
    def test_group_by_time(self):
        # We need to mock os.path.getmtime?
        # Or just create temp files?
        # Or modify PhotoGrouper to accept timestamps?
        # The method currently calls os.path.getmtime.
        # Let's patch os.path.getmtime.
        
        from unittest.mock import patch
        
        with patch('os.path.getmtime') as mock_mtime:
            # Setup: 3 files. A and B are close. C is far.
            # A: t=100
            # B: t=110 (diff 10s)
            # C: t=200 (diff 90s from B)
            
            mock_mtime.side_effect = lambda p: {
                'a.jpg': 100.0,
                'b.jpg': 110.0,
                'c.jpg': 200.0
            }.get(p, 0.0)
            
            images = ['a.jpg', 'b.jpg', 'c.jpg']
            events = PhotoGrouper.group_by_time(images, gap_threshold=60.0)
            
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0], ['a.jpg', 'b.jpg'])
            self.assertEqual(events[1], ['c.jpg'])

    def test_group_similar_simple(self):
        # A and B are similar (hash diff 0) and close in time
        # C is different
        
        images = [
            {'path': 'a', 'dhash': 'FFFF', 'timestamp': 100},
            {'path': 'b', 'dhash': 'FFFF', 'timestamp': 101},
            {'path': 'c', 'dhash': '0000', 'timestamp': 102}
        ]
        
        stacks = PhotoGrouper.group_similar(images, time_threshold=5)
        
        self.assertEqual(len(stacks), 2)
        # We can't guarantee order of stacks, but contents should be right
        # Sort stacks by first element path to verify
        stacks.sort(key=lambda x: x[0])
        
        self.assertEqual(stacks[0], ['a', 'b'])
        self.assertEqual(stacks[1], ['c'])

    def test_group_similar_transitive(self):
        # A similar to B, B similar to C. A NOT similar to C directly (distance > threshold).
        # But they should all be in one group because of B.
        
        # dHash 16 chars (64 bits). 
        # A: FFFF
        # B: FFFE (diff 1)
        # C: FFFC (diff 1 from B, 2 from A)
        
        images = [
            {'path': 'A', 'dhash': 'FFFF', 'timestamp': 100},
            {'path': 'B', 'dhash': 'FFFE', 'timestamp': 100},
            {'path': 'C', 'dhash': 'FFFC', 'timestamp': 100} 
        ]
        
        stacks = PhotoGrouper.group_similar(images, time_threshold=5)
        
        self.assertEqual(len(stacks), 1)
        self.assertEqual(set(stacks[0]), {'A', 'B', 'C'})
        
    def test_group_similar_time_break(self):
        # A and B identical hash, but far apart in time -> different stacks
        images = [
            {'path': 'A', 'dhash': 'FFFF', 'timestamp': 100},
            {'path': 'B', 'dhash': 'FFFF', 'timestamp': 200}
        ]
        
        stacks = PhotoGrouper.group_similar(images, time_threshold=10)
        self.assertEqual(len(stacks), 2)

    def test_group_similar_embeddings(self):
        # A and B: Sim = 1.0 (same embedding)
        # C: Sim = 0.0 (orthogonal)
        
        import numpy as np
        
        emb1 = np.array([1.0, 0.0])
        emb2 = np.array([1.0, 0.0]) # Sim with emb1 = 1.0
        emb3 = np.array([0.0, 1.0]) # Sim with emb1 = 0.0
        
        images = [
            {'path': 'A', 'embedding': emb1, 'timestamp': 100},
            {'path': 'B', 'embedding': emb2, 'timestamp': 100},
            {'path': 'C', 'embedding': emb3, 'timestamp': 100}
        ]
        
        # Threshold 0.85
        stacks = PhotoGrouper.group_similar(images, time_threshold=100, content_threshold=0.85)
        
        self.assertEqual(len(stacks), 2)
        # Sort to verify
        stacks.sort(key=lambda x: len(x), reverse=True)
        self.assertEqual(set(stacks[0]), {'A', 'B'})
        self.assertEqual(stacks[1], ['C'])

    def test_group_similar_fallback(self):
        # A and B have no embeddings, but close dHash
        images = [
            {'path': 'A', 'dhash': 'FFFF', 'timestamp': 100},
            {'path': 'B', 'dhash': 'FFFF', 'timestamp': 100}
        ]
        
        stacks = PhotoGrouper.group_similar(images, time_threshold=100)
        self.assertEqual(len(stacks), 1)
        self.assertEqual(set(stacks[0]), {'A', 'B'})

if __name__ == '__main__':
    unittest.main()
