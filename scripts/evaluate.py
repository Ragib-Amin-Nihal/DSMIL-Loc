#!/usr/bin/env python3
"""
Script for evaluating a trained whale call detection and localization model
"""

import torch
import numpy as np
from pathlib import Path
import json
import logging
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from datetime import datetime

from data.dataset import LocalizationDataset
from data.collate import collate_localization_bags
from models.mil_net import ImprovedLocalizationMILNet
from utils.visualization import ResultVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LocalizationEvaluator:
    def __init__(
        self,
        model_path: str,
        data_root: str,
        output_dir: str,
        device: str = 'cuda',
        temporal_tolerance: float = 15.0,
        attention_threshold: float = 0.02
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.temporal_tolerance = temporal_tolerance
        self.attention_threshold = attention_threshold
        
        # Load model
        self.model = self._load_model()
        
        # Initialize visualizer
        self.visualizer = ResultVisualizer(str(self.output_dir))
        
    def _load_model(self) -> torch.nn.Module:
        """Load trained model"""
        model = ImprovedLocalizationMILNet(feature_dim=512).to(self.device)
        
        logger.info(f"Loading model from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        return model
    
    def evaluate_site(self, site_years: list) -> dict:
        """Evaluate model performance on specific sites"""
        # Create dataset and loader
        dataset = LocalizationDataset(self.data_root, site_years)
        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_localization_bags
        )
        
        # Initialize metrics tracking
        all_metrics = {
            'detection': {
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'true_negatives': 0
            },
            'localization': {
                'temporal_errors': [],
                'detection_times': [],
                'ground_truth_times': []
            },
            'attention': {
                'weights': [],
                'peak_values': []
            }
        }
        
        # Evaluate batches
        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                    
                # Process batch
                batch_metrics = self._evaluate_batch(batch)
                
                # Update metrics
                self._update_metrics(all_metrics, batch_metrics, batch)
        
        # Compute final metrics
        final_metrics = self._compute_final_metrics(all_metrics)
        
        # Save results
        self._save_results(final_metrics, site_years)
        
        return final_metrics
    
    def _evaluate_batch(self, batch):
        """Evaluate single batch focusing on highest-attention instance per bag"""
        # Move data to device
        spectrograms = batch['spectrograms'].to(self.device)
        features = batch['features'].to(self.device)
        num_instances = batch['num_instances'].to(self.device)
        
        # Get model predictions
        outputs = self.model(spectrograms, features, num_instances)
        
        # Get attention weights
        attention_weights = outputs['attention_weights'].cpu().numpy()
        instance_timestamps = batch['instance_timestamps']
        
        batch_metrics = {
            'detection': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
            'localization': [],
            'attention': {
                'weights': attention_weights,
                'peak_values': []  # Track highest attention values
            }
        }
        
        # Process each bag in the batch
        for i in range(len(batch['bag_ids'])):
            # Get ground truth calls for this bag
            ground_truth_calls = batch['ground_truth_calls'][i]
            bag_label = batch['bag_labels'][i].item()
            
            # Get attention weights and timestamps for this bag
            bag_attention = attention_weights[i]
            bag_timestamps = instance_timestamps[i]
            
            # Find the instance with highest attention that exceeds threshold
            max_attention_idx = -1
            max_attention_value = -1
            
            for idx, attention in enumerate(bag_attention):
                if attention > self.attention_threshold and attention > max_attention_value:
                    max_attention_value = attention
                    max_attention_idx = idx
            
            # Record peak attention value
            batch_metrics['attention']['peak_values'].append(max_attention_value)
            
            # If no ground truth calls (negative bag)
            if bag_label == 0:
                if max_attention_idx == -1:
                    # True negative
                    batch_metrics['detection']['tn'] += 1
                    batch_metrics['localization'].append({
                        'bag_id': batch['bag_ids'][i],
                        'status': 'TN'
                    })
                else:
                    # False positive
                    batch_metrics['detection']['fp'] += 1
                    batch_metrics['localization'].append({
                        'bag_id': batch['bag_ids'][i],
                        'status': 'FP',
                        'detection_time': bag_timestamps[max_attention_idx],
                        'attention_value': max_attention_value
                    })
                continue
            
            # For positive bags
            if max_attention_idx == -1:
                # False negative
                batch_metrics['detection']['fn'] += 1
                batch_metrics['localization'].append({
                    'bag_id': batch['bag_ids'][i],
                    'status': 'FN',
                    'reason': 'No detection above threshold'
                })
                continue
            
            # Get timestamp of highest-attention instance
            detected_time = bag_timestamps[max_attention_idx]
            
            # Check if detection matches any ground truth call
            min_error = float('inf')
            matched = False
            
            for call in ground_truth_calls:
                call_time = call['start_time'] if isinstance(call['start_time'], (int, float)) else \
                    datetime.fromisoformat(call['start_time']).timestamp()
                
                error = abs(detected_time - call_time)
                if error <= self.temporal_tolerance and error < min_error:
                    min_error = error
                    matched = True
            
            # Update metrics based on matching
            if matched:
                batch_metrics['detection']['tp'] += 1
                batch_metrics['localization'].append({
                    'bag_id': batch['bag_ids'][i],
                    'status': 'TP',
                    'detection_time': detected_time,
                    'temporal_error': min_error,
                    'attention_value': max_attention_value
                })
            else:
                batch_metrics['detection']['fp'] += 1
                batch_metrics['localization'].append({
                    'bag_id': batch['bag_ids'][i],
                    'status': 'FP',
                    'detection_time': detected_time,
                    'attention_value': max_attention_value,
                    'reason': 'Detection outside temporal tolerance'
                })
        
        return batch_metrics
    
    def _update_metrics(self, all_metrics, batch_metrics, batch):
        """Update running metrics with batch results"""
        # Update detection counts
        all_metrics['detection']['true_positives'] += batch_metrics['detection']['tp']
        all_metrics['detection']['false_positives'] += batch_metrics['detection']['fp']
        all_metrics['detection']['false_negatives'] += batch_metrics['detection']['fn']
        all_metrics['detection']['true_negatives'] += batch_metrics['detection']['tn']
        
        # Update localization results
        for loc_result in batch_metrics['localization']:
            if loc_result.get('temporal_error') is not None:
                all_metrics['localization']['temporal_errors'].append(loc_result['temporal_error'])
            
            if loc_result.get('detection_time') is not None:
                all_metrics['localization']['detection_times'].append(loc_result['detection_time'])
        
        # Add ground truth times from all bags
        for i, gt_calls in enumerate(batch['ground_truth_calls']):
            for call in gt_calls:
                call_time = call['start_time'] if isinstance(call['start_time'], (int, float)) else \
                    datetime.fromisoformat(call['start_time']).timestamp()
                all_metrics['localization']['ground_truth_times'].append(call_time)
        
        # Store attention statistics
        all_metrics['attention']['weights'].extend(batch_metrics['attention']['weights'])
        all_metrics['attention']['peak_values'].extend(batch_metrics['attention']['peak_values'])
    
    def _compute_final_metrics(self, metrics):
        """Compute final metrics with focus on single-instance analysis"""
        # Detection metrics
        tp = metrics['detection']['true_positives']
        fp = metrics['detection']['false_positives']
        fn = metrics['detection']['false_negatives']
        tn = metrics['detection']['true_negatives']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # Localization metrics
        temporal_errors = metrics['localization']['temporal_errors']
        
        # Compile detailed metrics
        final_metrics = {
            'detection': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'true_negatives': tn
            },
            'localization': {
                'mean_temporal_error': np.mean(temporal_errors) if temporal_errors else 0,
                'median_temporal_error': np.median(temporal_errors) if temporal_errors else 0,
                'temporal_tolerance': self.temporal_tolerance,
                'max_temporal_error': np.max(temporal_errors) if temporal_errors else 0
            },
            'attention': {
                'mean_peak_attention': np.mean(metrics['attention']['peak_values']) if metrics['attention']['peak_values'] else 0,
                'threshold': self.attention_threshold
            },
            'raw_data': metrics
        }
        
        return final_metrics
    
    def _save_results(self, metrics, site_years):
        """Save evaluation results"""
        # Create site-specific output directory
        site_dir = self.output_dir / '_'.join(site_years)
        site_dir.mkdir(exist_ok=True)
        
        # Save metrics
        with open(site_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x)
        
        # Create
