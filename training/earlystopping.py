import numpy as np
from typing import Tuple, Dict

class TrendBasedEarlyStopping:
    def __init__(self, patience=15, window_size=10, min_epochs=20, min_improvement=0.01):
        self.patience = patience
        self.window_size = window_size
        self.min_epochs = min_epochs
        self.min_improvement = min_improvement
        self.f1_history = []
        self.best_f1 = 0
        self.best_epoch = 0
        
    def trend_analysis(self) -> Tuple[float, bool]:
        """
        Analyze trend in F1 scores over the window
        Returns:
            slope: trend direction and magnitude
            is_improving: whether the trend is positive enough
        """
        if len(self.f1_history) < self.window_size:
            return 0.0, True
            
        # Get recent scores
        recent_scores = self.f1_history[-self.window_size:]
        x = np.arange(len(recent_scores))
        
        # Compute trend line
        try:
            slope, _ = np.polyfit(x, recent_scores, 1)
        except Exception:
            slope = 0.0
        
        # Check if trend is improving significantly
        is_improving = slope > self.min_improvement
        
        return slope, is_improving
    
    def __call__(self, val_f1: float, epoch: int) -> Tuple[bool, Dict]:
        """
        Args:
            val_f1: Current validation F1 score
            epoch: Current epoch number
       
        Returns:
            should_stop: Whether training should stop
            info: Dictionary with trend information
        """
        self.f1_history.append(val_f1)
       
        # Don't stop before minimum epochs
        if epoch < self.min_epochs:
            return False, {
                'trend_slope': 0.0, 
                'is_improving': True, 
                'status': 'minimum_epochs_not_met',
                'epochs_since_best': 0,
                'best_f1': self.best_f1,
                'best_epoch': self.best_epoch,
                'current_f1': val_f1
            }
       
        # Update best score
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.best_epoch = epoch
       
        # Get trend
        slope, is_improving = self.trend_analysis()
       
        # Decision logic
        epochs_since_best = epoch - self.best_epoch
       
        info = {
            'trend_slope': float(slope),  # Ensure it's a float
            'is_improving': is_improving,
            'epochs_since_best': epochs_since_best,
            'best_f1': self.best_f1,
            'best_epoch': self.best_epoch,
            'current_f1': val_f1
        }
        
        # Stop if:
        # 1. Not improving over window AND haven't improved in patience epochs
        # 2. OR F1 score is zero/near-zero for too long
        # 3. BUT don't stop if we're still improving
        
        if len(self.f1_history) >= self.window_size:
            recent_mean = np.mean(self.f1_history[-5:])  # Last 5 epochs
           
            # Check for zero/near-zero F1 pattern
            if recent_mean < 0.01 and not is_improving:
                info['status'] = 'near_zero_f1'
                return True, info
           
            # Check for stagnation
            if not is_improving and epochs_since_best >= self.patience:
                info['status'] = 'stagnated'
                return True, info
       
        info['status'] = 'continuing'
        return False, info
