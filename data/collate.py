import torch
import numpy as np
from typing import Dict, List, Optional

def collate_whale_bags(batch):
    """Custom collate function for whale call bags"""
    if len(batch) == 0:
        return None
        
    # Find minimum number of instances across batch
    min_instances = min(b['num_instances'].item() for b in batch)
    
    # Truncate all bags to minimum number of instances
    processed_batch = []
    for item in batch:
        try:
            processed_batch.append({
                'bag_id': item['bag_id'],
                'spectrograms': item['spectrograms'][:min_instances],
                'features': item['features'][:min_instances],
                'bag_label': item['bag_label'],
                'num_instances': torch.tensor(min_instances),
                'site': item['site']
            })
        except Exception as e:
            print(f"Error processing batch item: {e}")
            continue
            
    if len(processed_batch) == 0:
        return None
        
    # Stack all tensors
    return {
        'bag_ids': [item['bag_id'] for item in processed_batch],
        'spectrograms': torch.stack([item['spectrograms'] for item in processed_batch]),
        'features': torch.stack([item['features'] for item in processed_batch]),
        'bag_labels': torch.stack([item['bag_label'] for item in processed_batch]),
        'num_instances': torch.stack([item['num_instances'] for item in processed_batch]),
        'sites': [item['site'] for item in processed_batch]
    }

def collate_localization_bags(batch):
    """Custom collate function for localization evaluation"""
    if len(batch) == 0:
        return None
        
    # Find minimum number of instances
    min_instances = min(b['num_instances'].item() for b in batch)
    
    # Process batch
    processed_batch = []
    for item in batch:
        try:
            # Build ground truth from instance labels
            instance_labels = torch.zeros(min_instances)
            instance_times = []
            ground_truth_calls = []
            
            # Process first min_instances instances
            for i in range(min_instances):
                # Store instance timestamp
                instance_times.append(item['instance_timestamps'][i])
                
                # Check if instance has any calls (non-empty labels)
                if item['instance_labels'][i]:
                    instance_labels[i] = 1
                    ground_truth_calls.append({
                        'start_time': item['instance_timestamps'][i],
                        'duration': 15.0  # Instance duration
                    })
            
            processed_batch.append({
                'bag_id': item['bag_id'],
                'spectrograms': item['spectrograms'][:min_instances],
                'features': item['features'][:min_instances],
                'bag_label': item['bag_label'],
                'num_instances': torch.tensor(min_instances),
                'site': item['site'],
                'instance_labels': instance_labels,
                'instance_timestamps': instance_times,
                'ground_truth_calls': ground_truth_calls
            })
            
        except Exception as e:
            print(f"Error processing batch item: {e}")
            continue
    
    if len(processed_batch) == 0:
        return None
    
    # Stack all tensors
    return {
        'bag_ids': [item['bag_id'] for item in processed_batch],
        'spectrograms': torch.stack([item['spectrograms'] for item in processed_batch]),
        'features': torch.stack([item['features'] for item in processed_batch]),
        'bag_labels': torch.stack([item['bag_label'] for item in processed_batch]),
        'num_instances': torch.stack([item['num_instances'] for item in processed_batch]),
        'sites': [item['site'] for item in processed_batch],
        'instance_labels': torch.stack([item['instance_labels'] for item in processed_batch]),
        'instance_timestamps': [item['instance_timestamps'] for item in processed_batch],
        'ground_truth_calls': [item['ground_truth_calls'] for item in processed_batch]
    }
