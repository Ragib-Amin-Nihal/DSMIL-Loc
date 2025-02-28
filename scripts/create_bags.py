#!/usr/bin/env python3
"""
Script to create bags from raw audio data in parallel
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path
import logging
import os
from typing import List
import argparse

from preprocessing.bag_creation import DynamicWhaleBagCreator

class BagCreationRunner:
    def __init__(
        self, 
        data_root: str, 
        output_root: str,
        site_years: List[str],
        world_size: int = None
    ):
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        self.site_years = site_years
        
        # Set up distributed processing
        self.world_size = world_size or torch.cuda.device_count()
        if self.world_size == 0:
            self.world_size = 1
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_distributed(self, rank: int):
        """Initialize distributed process group"""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        if self.world_size > 1:
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                init_method='env://',
                world_size=self.world_size,
                rank=rank
            )

            if torch.cuda.is_available():
                torch.cuda.set_device(rank)
            dist.barrier()
    
    def cleanup_distributed(self):
        """Cleanup distributed process group"""
        if self.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
    
    def create_bags(self, rank: int, bag_duration: int, instance_duration: int):
        """Create bags for all site years"""
        try:
            # Setup distributed process
            self.setup_distributed(rank)
            
            # Only create bags on rank 0 to avoid conflicts
            if rank == 0:
                self.logger.info(f"Creating bags with duration {bag_duration}s and instance duration {instance_duration}s")
                
                output_dir = self.output_root / f"results_bag{bag_duration}_inst{instance_duration}"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                bag_creator = DynamicWhaleBagCreator(
                    root_dir=self.data_root,
                    output_dir=output_dir,
                    bag_duration=bag_duration,
                    instance_duration=instance_duration
                )
                
                for site_year in self.site_years:
                    self.logger.info(f"Processing {site_year}")
                    bag_creator.process_site(site_year)
            
            # Wait for bag creation to complete
            if self.world_size > 1:
                dist.barrier()
            
        finally:
            self.cleanup_distributed()
    
    def create_all_configurations(self, bag_durations, instance_durations):
        """Create bags for all duration configurations"""
        for bag_duration in bag_durations:
            for instance_duration in instance_durations:
                if instance_duration < bag_duration:
                    if self.world_size > 1:
                        # Launch distributed processes for bag creation
                        mp.spawn(
                            self.create_bags,
                            args=(bag_duration, instance_duration),
                            nprocs=self.world_size,
                            join=True
                        )
                    else:
                        # Single process mode
                        self.create_bags(0, bag_duration, instance_duration)

def parse_args():
    parser = argparse.ArgumentParser(description='Create whale call bags from raw audio data')
    
    parser.add_argument('--data-root', type=str, required=True,
                       help='Root directory containing the raw dataset')
    
    parser.add_argument('--output-root', type=str, required=True,
                       help='Root directory for processed data output')
    
    parser.add_argument('--site-years', type=str, nargs='+', required=True,
                       help='List of site years to process')
    
    parser.add_argument('--bag-durations', type=int, nargs='+', default=[300],
                       help='List of bag durations in seconds')
    
    parser.add_argument('--instance-durations', type=int, nargs='+', default=[15],
                       help='List of instance durations in seconds')
    
    parser.add_argument('--single-process', action='store_true',
                       help='Run in single process mode instead of using distributed processing')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get number of available GPUs
    world_size = 1 if args.single_process else torch.cuda.device_count()
    if world_size == 0:
        world_size = 1  # Use CPU if no GPUs available
        
    print(f"Running with {world_size} {'processes' if world_size > 1 else 'process'}")
    
    # Create and run bag creator
    runner = BagCreationRunner(
        data_root=args.data_root,
        output_root=args.output_root,
        site_years=args.site_years,
        world_size=world_size
    )
    
    # Create bags for all configurations
    runner.create_all_configurations(args.bag_durations, args.instance_durations)
    
    print("Bag creation complete!")

if __name__ == "__main__":
    main()
