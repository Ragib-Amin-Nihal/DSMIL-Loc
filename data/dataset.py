import torch
from torch.utils.data import Dataset
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import concurrent.futures
from functools import partial
import torch.nn.functional as F

class LocalizationDataset(Dataset):
    def __init__(self, data_dir: str, site_years: Optional[List[str]] = None,
                 cache_size: int = 1000000, num_workers: int = 8, 
                 preload_data: bool = False):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.site_years = site_years if site_years else self._get_available_site_years()
        self.cache_size = cache_size
        self.num_workers = num_workers
        
        # Define feature keys
        self.temporal_feature_keys = [
            'rms_energy', 'peak_amplitude', 'zero_crossing_rate',
            'envelope_mean', 'envelope_std', 'skewness',
            'kurtosis', 'temporal_centroid'
        ]
        
        self.physics_feature_keys = [
            'snr', 'source_intensity', 'spectral_slope',
            'Bm-Ant-A_energy_ratio', 'Bm-Ant-B_energy_ratio',
            'Bm-Ant-C_energy_ratio', 'Bp-20Hz_energy_ratio',
            'Bp-High_energy_ratio'
        ]
        
        # Initialize cache
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Load all bags
        self.bags = self._load_all_bags()
        
        # Preload data if requested
        if preload_data:
            self._preload_data()
            
        self._print_statistics()

    def __len__(self):
        """Return the number of bags in the dataset"""
        return len(self.bags)

    def _get_available_site_years(self) -> List[str]:
        """Get all available site years from the data directory"""
        return [d.name for d in self.data_dir.iterdir() if d.is_dir()]

    def load_instance_parallel(self, bag_dir: Path, i: int, target_shape=(129, 235)):
        """Load single instance data in parallel"""
        try:
            # Load spectrogram
            spec_path = bag_dir / 'spectrograms' / f'instance_{i:04d}_spec.npy'
            spectrogram = np.load(spec_path, mmap_mode='r')  # Memory mapped loading

            # Handle dimensions
            if spectrogram.shape != target_shape:
                if spectrogram.shape[1] > target_shape[1]:
                    spectrogram = spectrogram[:, :target_shape[1]]
                else:
                    pad_width = ((0, 0), (0, target_shape[1] - spectrogram.shape[1]))
                    spectrogram = np.pad(spectrogram, pad_width, mode='constant')

            # Load features
            feature_path = bag_dir / 'features' / f'instance_{i:04d}_features.json'
            with open(feature_path, 'r') as f:
                instance_data = json.load(f)

            # Extract feature vector
            feature_vector = []
            for key in self.temporal_feature_keys:
                feature_vector.append(float(instance_data['temporal_features'].get(key, 0.0)))
            for key in self.physics_feature_keys:
                feature_vector.append(float(instance_data['physics_features'].get(key, 0.0)))

            return i, spectrogram, feature_vector, instance_data.get('start_time'), instance_data.get('labels', [])

        except Exception as e:
            print(f"Error loading instance {i}: {e}")
            return None

    def _load_all_bags(self) -> List[Dict]:
        """Load metadata for all bags"""
        all_bags = []
        
        # Look in the 'bags' directory
        bags_dir = self.data_dir / 'bags'
        
        # Iterate through site years
        for site_year in self.site_years:
            site_dir = bags_dir / site_year
            
            if not site_dir.exists():
                print(f"Warning: Site year directory not found: {site_dir}")
                continue

            # Find all bag directories
            bag_dirs = sorted([d for d in site_dir.iterdir() if d.is_dir()])
            print(f"Found {len(bag_dirs)} bags in {site_year}")

            for bag_dir in bag_dirs:
                try:
                    # Load metadata
                    with open(bag_dir / 'metadata.json', 'r') as f:
                        metadata = json.load(f)
                    
                    # Add bag directory and site year
                    metadata['bag_dir'] = str(bag_dir)
                    metadata['site_year'] = site_year
                    
                    # Bag label is directly available from has_calls
                    metadata['bag_label'] = 1 if metadata['has_calls'] else 0
                    
                    all_bags.append(metadata)
                    
                except Exception as e:
                    print(f"Error loading metadata from {bag_dir}: {e}")
                    continue

        return all_bags

    def __getitem__(self, idx):
        """Get bag with parallel loading and caching"""
        bag = self.bags[idx]
        bag_dir = Path(bag['bag_dir'])
        
        # Prepare parallel loading
        load_func = partial(self.load_instance_parallel, bag_dir)
        instance_indices = range(bag['n_instances'])
        
        # Initialize storage
        spectrograms = []
        features = []
        instance_timestamps = []
        instance_labels = []
        
        # Use ThreadPoolExecutor instead of ProcessPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all loading tasks
            futures = [executor.submit(load_func, i) for i in instance_indices]
            
            # Process results as they complete 
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    i, spec, feat, start_time, labels = result
                    spectrograms.append(spec)
                    features.append(feat)
                    
                    # Calculate timestamp
                    if isinstance(start_time, str):
                        instance_time = datetime.fromisoformat(start_time)
                        instance_timestamps.append(instance_time.timestamp())
                    else:
                        # Fallback if start_time is not available
                        base_time = datetime.fromisoformat(bag['start_time'])
                        instance_duration = bag['duration'] / bag['n_instances']
                        instance_time = base_time + timedelta(seconds=i * instance_duration)
                        instance_timestamps.append(instance_time.timestamp())
                    
                    # Set instance label (1 if has calls, 0 otherwise)
                    instance_labels.append(1 if labels else 0)
        
        if len(spectrograms) == 0:
            raise ValueError(f"No valid instances found in bag {bag['bag_id']}")
        
        # Create instance labels if not already set
        if not instance_labels:
            instance_labels = np.zeros(bag['n_instances'])
            if bag['has_calls']:
                for call_info in bag['instances_with_calls']:
                    instance_labels[call_info['instance_idx']] = 1
        
        return {
            'bag_id': bag['bag_id'],
            'spectrograms': torch.FloatTensor(np.stack(spectrograms)),
            'features': torch.FloatTensor(np.array(features)),
            'bag_label': torch.tensor(bag['bag_label'], dtype=torch.float32),
            'num_instances': torch.tensor(len(spectrograms), dtype=torch.long),
            'site': bag['site_year'],
            'instance_timestamps': instance_timestamps,
            'instance_labels': torch.FloatTensor(instance_labels),
            'instances_with_calls': bag.get('instances_with_calls', [])
        }

    def _print_statistics(self):
        """Print detailed dataset statistics"""
        total_bags = len(self.bags)
        positive_bags = sum(1 for bag in self.bags if bag['has_calls'])
        total_instances = sum(bag['n_instances'] for bag in self.bags)
        total_instances_with_calls = sum(
            len(bag['instances_with_calls']) 
            for bag in self.bags if bag['has_calls']
        )
        
        print("\nDataset Statistics:")
        print(f"Total bags: {total_bags}")
        print(f"Positive bags: {positive_bags} ({positive_bags/total_bags*100:.1f}%)")
        print(f"Negative bags: {total_bags - positive_bags} ({(total_bags-positive_bags)/total_bags*100:.1f}%)")
        print(f"Total instances: {total_instances}")
        print(f"Instances with calls: {total_instances_with_calls} ({total_instances_with_calls/total_instances*100:.1f}%)")
        
        # Print statistics per site year
        print("\nPer Site Statistics:")
        site_stats = {}
        for bag in self.bags:
            site = bag['site_year']
            if site not in site_stats:
                site_stats[site] = {'total': 0, 'positive': 0, 'instances_with_calls': 0}
            site_stats[site]['total'] += 1
            if bag['has_calls']:
                site_stats[site]['positive'] += 1
                site_stats[site]['instances_with_calls'] += len(bag['instances_with_calls'])
        
        for site, stats in site_stats.items():
            print(f"\n{site}:")
            print(f"  Total bags: {stats['total']}")
            print(f"  Positive bags: {stats['positive']} ({stats['positive']/stats['total']*100:.1f}%)")
            print(f"  Instances with calls: {stats['instances_with_calls']}")

    def _preload_data(self):
        """Preload all data into memory with progress bar"""
        print("\nPreloading data into memory...")
        from tqdm import tqdm
        import psutil
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for idx in range(len(self)):
                bag = self.bags[idx]
                bag_dir = Path(bag['bag_dir'])
                for i in range(bag['n_instances']):
                    futures.append(executor.submit(self.load_instance_parallel, bag_dir, i))
                
            for _ in tqdm(concurrent.futures.as_completed(futures), 
                         total=len(futures), 
                         desc="Loading data"):
                pass
                
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.memory_usage = final_memory - initial_memory
        
        print(f"\nMemory usage for preloaded data: {self.memory_usage:.2f} MB")
