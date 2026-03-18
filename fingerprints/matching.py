import numpy as np

def match_minutiae(set_a: list, set_b: list, dist_tol: int = 15, angle_tol: float = 0.2) -> float:
    """
    Simple alignment-based matcher. Returns a score between 0 and 1.
    """
    matches = 0
    # Create a copy or a flag array to mark used minutiae in set_b if we want to ensure 1-to-1 matching
    # For now, simplistic approach as per existing code.
    
    # We should probably track which indices in set_b are used to avoid many-to-one mapping
    used_indices_b = set()

    for m1 in set_a:
        best_dist = float('inf')
        best_idx = -1
        
        for i, m2 in enumerate(set_b):
            if i in used_indices_b:
                continue
            
            d = np.sqrt((m1['x'] - m2['x'])**2 + (m1['y'] - m2['y'])**2)
            # Angle difference handling periodicity
            a_diff = abs(m1['angle'] - m2['angle'])
            if a_diff > np.pi:
                a_diff = 2 * np.pi - a_diff
            
            if d < dist_tol and a_diff < angle_tol and m1['type'] == m2['type']:
                if d < best_dist:
                    best_dist = d
                    best_idx = i
        
        if best_idx != -1:
            matches += 1
            used_indices_b.add(best_idx)

    return matches / max(len(set_a), len(set_b), 1)
