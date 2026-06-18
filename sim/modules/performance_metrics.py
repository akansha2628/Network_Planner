"""
Performance Metrics Module

Tracks and computes system performance metrics such as:
- Blocking probability
- Link utilization
- Throughput
"""

from typing import Dict, List, Tuple
from collections import defaultdict


class PerformanceMetrics:
    """Tracks and computes network performance metrics."""
    
    def __init__(self):
        """Initialize performance metrics tracking."""
        self.reset()
    
    def reset(self):
        """Reset all metrics to initial state."""
        self.connection_count = 0
        self.blocked_connection_count = 0
        self.arrived_datarate = 0.0
        self.accepted_datarate = 0.0
        self.blocked_datarate = 0.0
        
        # Time series data
        self.time_points = []
        self.blocking_prob_points = []
        
        # Per source-destination metrics
        self.sd_arrivals = defaultdict(int)
        self.sd_blocked = defaultdict(int)
        
    def record_connection(
        self,
        src: int,
        dest: int,
        datarate: float,
        blocked: bool,
        current_time: float
    ):
        """
        Record a connection attempt.
        
        Args:
            src: Source node
            dest: Destination node
            datarate: Connection datarate in Gbps
            blocked: Whether connection was blocked
            current_time: Current simulation time
        """
        self.connection_count += 1
        self.arrived_datarate += datarate
        
        # Create undirected SD pair key
        sd_key = (src, dest) if src < dest else (dest, src)
        self.sd_arrivals[sd_key] += 1
        
        if blocked:
            self.blocked_connection_count += 1
            self.blocked_datarate += datarate
            self.sd_blocked[sd_key] += 1
        else:
            self.accepted_datarate += datarate
        
        # Record blocking probability at this time
        if self.connection_count > 0:
            blocking_prob = self.blocked_connection_count / self.connection_count
            self.time_points.append(current_time)
            self.blocking_prob_points.append(blocking_prob)
    
    def get_blocking_probability(self) -> float:
        """
        Get overall blocking probability.
        
        Returns:
            Blocking probability (0.0 to 1.0)
        """
        if self.connection_count == 0:
            return 0.0
        return self.blocked_connection_count / self.connection_count
    
    def get_sd_blocking_probability(self) -> Dict[Tuple[int, int], float]:
        """
        Get per source-destination blocking probabilities.
        
        Returns:
            Dictionary mapping (src, dest) pairs to blocking probability
        """
        sd_block_prob = {}
        for sd in self.sd_arrivals:
            if self.sd_arrivals[sd] > 0:
                sd_block_prob[sd] = self.sd_blocked[sd] / self.sd_arrivals[sd]
            else:
                sd_block_prob[sd] = 0.0
        return sd_block_prob
    
    def get_throughput(self) -> float:
        """
        Get network throughput (accepted datarate / arrived datarate).
        
        Returns:
            Throughput ratio (0.0 to 1.0)
        """
        if self.arrived_datarate == 0:
            return 0.0
        return self.accepted_datarate / self.arrived_datarate
    
    def get_critical_sd_pairs(self, threshold: float) -> List[Tuple[int, int]]:
        """
        Get source-destination pairs with blocking above threshold.
        
        Args:
            threshold: Blocking probability threshold
            
        Returns:
            List of (src, dest) pairs exceeding threshold
        """
        sd_block_prob = self.get_sd_blocking_probability()
        critical_pairs = [
            sd for sd, bp in sd_block_prob.items()
            if bp >= threshold
        ]
        return critical_pairs
    
    def print_summary(self):
        """Print performance metrics summary."""
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)
        print(f"Total connections: {self.connection_count}")
        print(f"Blocked connections: {self.blocked_connection_count}")
        print(f"Blocking probability: {self.get_blocking_probability():.4f}")
        print(f"\nDatarate:")
        print(f"  Arrived: {self.arrived_datarate:.2f} Gbps")
        print(f"  Accepted: {self.accepted_datarate:.2f} Gbps")
        print(f"  Blocked: {self.blocked_datarate:.2f} Gbps")
        print(f"  Throughput: {self.get_throughput():.4f}")
        print("="*60 + "\n")
