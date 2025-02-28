import torch
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json
from tqdm import tqdm
from scipy import signal, stats
import warnings
warnings.filterwarnings('ignore')
from typing import List, Tuple, Optional
import os
import concurrent.futures

class DynamicWhaleBagCreator:
    def __init__(
        self, 
        root_dir: str, 
        output_dir: str, 
        bag_duration: int = 300,  # Default 5 minutes
        instance_duration: int = 15,  # Default 15 seconds
        instance_overlap: int = 0  # No overlap
    ):
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dynamic durations
        self.bag_duration = bag_duration
        self.instance_duration = instance_duration
        self.instance_overlap = instance_overlap
        
        # Audio parameters 
        self.target_sr = 250
        
        # Calculate samples
        self.samples_per_bag = int(self.bag_duration * self.target_sr)
        self.samples_per_instance = int(self.instance_duration * self.target_sr)
        self.overlap_samples = int(self.instance_overlap * self.target_sr)
        
        # Spectrogram parameters
        self.nfft = 256
        self.hop_length = self.nfft // 16
        self.win_length = self.nfft
        self.freq_range = (5, 124)  # Hz
        
        # Call frequency bands (kept from original implementation)
        self.call_bands = {
            'Bm-Ant-A': (25, 28),
            'Bm-Ant-B': (19.5, 28),
            'Bm-Ant-C': (18.5, 19),
            'Bp-20Hz': (15, 30),
            'Bp-High': (80, 100)
        }
    
    def create_instances_from_audio(self, audio: np.ndarray, 
                              start_time: datetime,
                              annotations_df: pd.DataFrame,
                              site_year: str) -> list:
        """Create instances from audio segment with dynamic duration"""
        instances = []
        samples_per_instance = int(self.instance_duration * self.target_sr)
        overlap_samples = int(self.instance_overlap * self.target_sr)
        step = samples_per_instance - overlap_samples
        
        # Process each instance
        for inst_idx, inst_start in enumerate(range(0, len(audio), step)):
            try:
                inst_end = inst_start + samples_per_instance
                if inst_end > len(audio):
                    break
                    
                # Get audio segment
                instance_audio = audio[inst_start:inst_end]
                instance_start_time = start_time + timedelta(seconds=inst_start/self.target_sr)
                
                # Get freqs for feature extraction
                freqs = librosa.fft_frequencies(sr=self.target_sr, n_fft=self.nfft)
                
                # Extract all features
                spec_features = self.extract_spectrogram_features(instance_audio)
                temporal_features = self.extract_temporal_features(instance_audio)
                physics_features = self.extract_physics_features(
                    instance_audio, 
                    spec_features['spec_mag'],
                    freqs
                )
                
                # Get labels
                labels = self._get_instance_labels(
                    instance_start_time,
                    instance_start_time + timedelta(seconds=self.instance_duration),
                    annotations_df
                )
                
                # Create instance metadata
                instance_data = {
                    'id': f"{inst_idx:04d}",  # Use original index
                    'start_time': instance_start_time.isoformat(),
                    'duration': self.instance_duration,
                    'labels': labels,
                    'temporal_features': temporal_features,
                    'physics_features': physics_features,
                    'band_energies': spec_features['band_energies']
                }
                
                # Store instance
                instances.append({
                    'metadata': instance_data,
                    'spectrogram': spec_features['spectrogram']
                })
                    
            except Exception as e:
                print(f"Error processing instance {inst_idx}: {e}")
                continue
        
        return instances

    def save_bag(self, bag_id: str, instances: list, output_dir: Path, site_year: str):
        """Save bag data with organized structure"""
        try:
            # Create complete directory structure
            bags_dir = output_dir / 'bags' / site_year / bag_id
            bags_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            spec_dir = bags_dir / 'spectrograms'
            feature_dir = bags_dir / 'features'
            spec_dir.mkdir(exist_ok=True)
            feature_dir.mkdir(exist_ok=True)
            
            # Track call presence and instances with calls
            has_calls = False
            instances_with_calls = []
            
            # Save instances
            for instance in instances:
                inst_id = instance['metadata']['id']
                
                try:
                    # Save spectrogram
                    spec_path = spec_dir / f'instance_{inst_id}_spec.npy'
                    np.save(spec_path, instance['spectrogram'])
                    
                    # Save features
                    feat_path = feature_dir / f'instance_{inst_id}_features.json'
                    with open(feat_path, 'w') as f:
                        json.dump(instance['metadata'], f, indent=2)
                    
                    # Check for calls
                    if instance['metadata']['labels']:
                        has_calls = True
                        instances_with_calls.append({
                            'instance_idx': int(inst_id),
                            'labels': instance['metadata']['labels']
                        })
                        
                except Exception as e:
                    print(f"Error saving instance {inst_id}: {e}")
            
            # Save bag metadata
            metadata_path = bags_dir / 'metadata.json'
            metadata = {
                'bag_id': bag_id,
                'site_year': site_year,
                'n_instances': len(instances),
                'start_time': instances[0]['metadata']['start_time'],
                'end_time': instances[-1]['metadata']['start_time'],
                'duration': self.bag_duration,
                'has_calls': has_calls,
                'instances_with_calls': instances_with_calls,
                'parameters': {
                    'sampling_rate': self.target_sr,
                    'nfft': self.nfft,
                    'hop_length': self.hop_length,
                    'win_length': self.win_length,
                    'freq_range': self.freq_range
                }
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"Error in save_bag for {bag_id}: {e}")
            raise

    def extract_spectrogram_features(self, audio: np.ndarray) -> dict:
        """Extract spectrogram and related features"""
        n_fft = 256
        hop_length = n_fft // 16
        win_length = n_fft
        # Compute spectrogram
        spec = librosa.stft(
            audio,
            n_fft=n_fft,
            hop_length= hop_length,
            win_length= win_length,
            window='hann'
        )
        spec_mag = np.abs(spec)
        spec_db = librosa.amplitude_to_db(spec_mag)
        
        # Get frequency bins
        freqs = librosa.fft_frequencies(sr=self.target_sr, n_fft=n_fft)
        
        # Extract band energies
        band_energies = {}
        for band_name, (low, high) in self.call_bands.items():
            mask = (freqs >= low) & (freqs <= high)
            band_energies[band_name] = float(np.mean(spec_db[mask]))
        
        return {
            'spectrogram': spec_db,
            'band_energies': band_energies,
            'spec_mag': spec_mag
        }

    def extract_temporal_features(self, audio: np.ndarray) -> dict:
        """Extract temporal domain features"""
        features = {}
        
        # Basic statistics
        features['rms_energy'] = float(np.sqrt(np.mean(audio**2)))
        features['peak_amplitude'] = float(np.max(np.abs(audio)))
        features['zero_crossing_rate'] = float(librosa.feature.zero_crossing_rate(audio)[0].mean())
        
        # Energy envelope
        envelope = np.abs(signal.hilbert(audio))
        features['envelope_mean'] = float(np.mean(envelope))
        features['envelope_std'] = float(np.std(envelope))
        
        # Statistical moments
        features['skewness'] = float(stats.skew(audio))
        features['kurtosis'] = float(stats.kurtosis(audio))
        
        # Temporal centroid
        energy_envelope = envelope**2
        times = np.arange(len(audio)) / self.target_sr
        features['temporal_centroid'] = float(np.sum(times * energy_envelope) / np.sum(energy_envelope))
        
        return features

    def extract_physics_features(self, audio: np.ndarray, spec_mag: np.ndarray, freqs: np.ndarray) -> dict:
        """Extract physics-based features"""
        features = {}
        
        # 1. SNR Estimation
        noise_floor = np.percentile(spec_mag, 10, axis=1)
        signal_power = np.max(spec_mag, axis=1)
        features['snr'] = float(np.mean(20 * np.log10(signal_power / (noise_floor + 1e-10))))
        
        # 2. Frequency Modulation
        # Track dominant frequency over time
        dominant_freqs = freqs[np.argmax(spec_mag, axis=0)]
        features['mod_rate'] = float(np.mean(np.abs(np.diff(dominant_freqs))))
        features['mod_std'] = float(np.std(np.diff(dominant_freqs)))
        
        # 3. Source Characteristics
        # Estimate source intensity considering propagation
        rms_energy = np.sqrt(np.mean(audio**2))
        peak_amplitude = np.max(np.abs(audio))
        features['source_intensity'] = float(rms_energy * np.log10(peak_amplitude + 1e-10))
        
        # 4. Spectral Slope and Shape
        spec_mean = np.mean(spec_mag, axis=1)
        slope, _ = np.polyfit(np.arange(len(spec_mean)), spec_mean, 1)
        features['spectral_slope'] = float(slope)
        
        # 5. Energy Distribution
        total_energy = np.sum(spec_mag**2)
        for band_name, (low, high) in self.call_bands.items():
            mask = (freqs >= low) & (freqs <= high)
            band_energy = np.sum(spec_mag[mask]**2)
            features[f'{band_name}_energy_ratio'] = float(band_energy / total_energy)
        
        return features

    def _get_instance_labels(self, start_time: datetime, end_time: datetime, 
                        annotations_df: pd.DataFrame) -> List[str]:
        """Get call types occurring within a time window"""
        try:
            # Find all annotations that overlap with the instance time window
            overlapping = annotations_df[
                (annotations_df['Begin Date Time'] <= end_time) & 
                (annotations_df['Begin Date Time'] + pd.to_timedelta(annotations_df['Delta Time (s)'], unit='s') >= start_time)
            ]
            
            # Return unique call types present
            if len(overlapping) > 0:
                return overlapping['call_type'].unique().tolist()
            return []
            
        except Exception as e:
            print(f"Error getting labels for time window {start_time} to {end_time}: {e}")
            return []

    def _load_annotations(self, site_year: str) -> pd.DataFrame:
        """Load and parse annotation files for a site year"""
        try:
            site_dir = self.root_dir / site_year
            all_selections = []
            
            # Find all selection files using multiple patterns
            selection_files = []
            for pattern in ['*.selections.txt', '*.txt']:
                selection_files.extend(list(site_dir.glob(pattern)))
            
            # Filter for relevant whale call files
            selection_files = [f for f in selection_files if any(
                call_type in f.stem for call_type in ['Bm', 'Bp', 'Fin', 'Blue']
            )]
            
            if not selection_files:
                print(f"No annotation files found for {site_year}")
                return None
                
            print(f"Found {len(selection_files)} annotation files")
            
            # Process each selection file
            for sel_file in selection_files:
                try:
                    # Read selection table
                    df = pd.read_csv(sel_file, delimiter='\t')
                    
                    # Get call type from filename using improved pattern matching
                    call_type = None
                    filename = sel_file.stem.lower()  # Case-insensitive matching
                    
                    # 1. Antarctic Blue Whale Calls
                    if any(x in filename for x in ['bm.ant-', 'bmant-', 'bm_ant']):
                        if any(x in filename for x in ['a', '-a.', '-a_']):
                            call_type = 'Bm-Ant-A'
                        elif 'b' in filename:
                            call_type = 'Bm-Ant-B'
                        elif 'z' in filename:
                            call_type = 'Bm-Ant-Z'
                            
                    # 2. Blue Whale D Calls    
                    elif any(x in filename for x in ['bm.d', 'bmd', 'dcalls']):
                        call_type = 'Bm-D'
                        
                    # 3. Fin Whale Calls
                    elif '20plus' in filename.replace('.', '').replace('-', '').replace('_', ''):
                        call_type = 'Bp-20Plus'
                    elif '20hz' in filename.replace('.', '').replace('-', '').replace('_', ''):
                        call_type = 'Bp-20Hz'
                    elif any(x in filename for x in ['downsweep', 'dwnswp', '.ds.', '_ds_']):
                        call_type = 'Bp-Downsweep'
                    
                    if call_type is None:
                        print(f"Unknown call type in file: {sel_file.name}")
                        continue
                    
                    # Add required columns
                    df['call_type'] = call_type
                    df['Begin Date Time'] = pd.to_datetime(df['Begin Date Time'])
                    df['End Date Time'] = df['Begin Date Time'] + pd.to_timedelta(df['Delta Time (s)'], unit='s')
                    
                    all_selections.append(df)
                    print(f"Loaded {len(df)} annotations of type {call_type} from {sel_file.name}")
                    
                except Exception as e:
                    print(f"Error loading {sel_file}: {e}")
                    continue
            
            if not all_selections:
                return None
                
            # Combine all annotations
            annotations_df = pd.concat(all_selections, ignore_index=True)
            print(f"Total annotations loaded: {len(annotations_df)}")
            
            return annotations_df
            
        except Exception as e:
            print(f"Error loading annotations for {site_year}: {e}")
            return None
            
    def _parse_timestamp(self, filename: str) -> datetime:
        """Parse timestamp from audio filename"""
        try:
            basename = Path(filename).stem

            # Handle case starting with 200_ or 201_
            if basename.startswith(('200_', '201_')):
                # For formats like: 200_2013-12-25_06-00-00 and 201_2014-02-22_05-00-00
                parts = basename.split('_')
                if len(parts) >= 3:
                    date_str = parts[1]  # '2013-12-25'
                    time_str = parts[2]  # '06-00-00'
                    return datetime.strptime(f"{date_str}_{time_str}", '%Y-%m-%d_%H-%M-%S')

            # Handle format with dash
            elif '-' in basename and basename.startswith('20'):
                # For format like: 20150102-140944
                return datetime.strptime(basename, '%Y%m%d-%H%M%S')

            # Handle standard format (YYYYMMDD_HHMMSS) with optional extras
            elif '_' in basename and basename.startswith('20'):
                # For all other formats starting with 20 and containing underscore
                parts = basename.split('_')
                if len(parts) >= 2:
                    date_part = parts[0]    # YYYYMMDD
                    time_part = parts[1]    # HHMMSS
                    return datetime.strptime(f"{date_part}{time_part}", '%Y%m%d%H%M%S')

            raise ValueError(f"Unknown timestamp format: {basename}")

        except Exception as e:
            print(f"Error parsing timestamp from {filename}: {e}")
            return None
            
    def _process_single_file(self, audio_file: Path, annotations_df: pd.DataFrame,
                        site_year: str, site_dir: Path):
        """Process single audio file for parallel execution"""
        try:
            # Load audio
            audio, sr = librosa.load(str(audio_file), sr=self.target_sr)
            
            # Get timestamp
            start_time = self._parse_timestamp(audio_file.stem)
            if start_time is None:
                return
            
            # Process into bags
            self._process_audio_file(
                audio=audio,
                start_time=start_time,
                annotations_df=annotations_df,
                site_year=site_year,
                site_dir=site_dir
            )
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")

    def process_site(self, site_year: str):
        """Process all files for a site in parallel"""
        print(f"\nProcessing {site_year}...")
        
        # Create output directory
        site_dir = self.output_dir / site_year
        site_dir.mkdir(parents=True, exist_ok=True)
        
        # Load annotations
        annotations_df = self._load_annotations(site_year)
        if annotations_df is None:
            return
        
        # Process audio files in parallel
        wav_dir = self.root_dir / site_year / 'wav'
        audio_files = sorted(list(wav_dir.glob('*.wav')))
        
        # Use ProcessPoolExecutor for CPU-intensive tasks
        max_workers = int(os.cpu_count() * 0.80) if os.cpu_count() else 4
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for audio_file in audio_files:
                future = executor.submit(
                    self._process_single_file,
                    audio_file=audio_file,
                    annotations_df=annotations_df,
                    site_year=site_year,
                    site_dir=site_dir
                )
                futures.append(future)
            
            # Process results with progress bar
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Processing audio files"
            ):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing file: {e}")

    def _process_audio_file(self, audio: np.ndarray, start_time: datetime,
                       annotations_df: pd.DataFrame, site_year: str,
                       site_dir: Path):
        """Process single audio file into bags"""
        samples_per_bag = int(self.bag_duration * self.target_sr)
        
        # Add logging
        # print(f"Processing audio file starting at {start_time}")
        bags_created = 0
        
        for bag_idx, bag_start in enumerate(range(0, len(audio), samples_per_bag)):
            bag_end = bag_start + samples_per_bag
            if bag_end > len(audio):
                break
                
            # Get bag audio
            bag_audio = audio[bag_start:bag_end]
            bag_start_time = start_time + timedelta(seconds=bag_idx*self.bag_duration)
            
            # Create instances
            instances = self.create_instances_from_audio(
                bag_audio,
                bag_start_time,
                annotations_df,
                site_year
            )
            
            # Save bag if enough valid instances
            if len(instances) >= 2:
                try:
                    # Create unique bag ID including timestamp
                    bag_id = f"{bag_start_time.strftime('%Y%m%d_%H%M%S')}"
                    
                    # Save the bag
                    self.save_bag(bag_id, instances, self.output_dir, site_year)
                    bags_created += 1
                    
                except Exception as e:
                    print(f"Error saving bag {bag_id}: {e}")
