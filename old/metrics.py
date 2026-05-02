from typing import Set
import math

def calculate_jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """
    Calculates Jaccard Similarity between two sets of strings (KPI names).
    Formula: |Intersection| / |Union|
    """
    if not set_a and not set_b:
        return 0.0 # Both empty implies no SLA overlap, essentially
    
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    
    if union == 0:
        return 0.0
    
    return intersection / union

def calculate_joint_decision_space(size_a: int, size_b: int) -> int:
    """
    Calculates the size of the joint decision space.
    """
    return size_a * size_b

def calculate_complexity_penalty(joint_size: int, soft_limit: int = 100) -> float:
    """
    Normalizes the complexity into a 0-1 penalty score.
    Uses a simple logistical or linear saturation function.
    Here, a linear ramp up to the limit.
    """
    if joint_size <= 1:
        return 0.0
    
    penalty = joint_size / soft_limit
    return min(penalty, 1.0)

def calculate_edge_weight(
    reachable_a: Set[str], 
    reachable_b: Set[str], 
    size_a: int, 
    size_b: int,
    alpha: float = 1.0, 
    beta: float = 0.5,
    complexity_limit: int = 100
) -> float:
    """
    Calculates the final weight for the edge between two tasks.
    Weight = (alpha * Pull) - (beta * Push)
    Clamped to 0.
    """
    pull = calculate_jaccard_similarity(reachable_a, reachable_b)
    
    joint_size = calculate_joint_decision_space(size_a, size_b)
    push = calculate_complexity_penalty(joint_size, complexity_limit)
    
    weight = (alpha * pull) - (beta * push)
    
    return max(0.0, weight)
