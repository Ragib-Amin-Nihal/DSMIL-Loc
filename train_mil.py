#!/usr/bin/env python3
"""
Script for distributed training of whale call detection models across multiple GPUs
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path
import socket
import random
import time
from contextlib import closing
import json
import logging
import os
import argparse
from typing import List, Dict, Any, Optional

from preprocessing.bag_creation import DynamicWhaleBagCreator
from utils.config_loader import ConfigLoader, get_default_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types and PyTorch tensors"""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)

def save_json(data: Any, filepath: str, **kwargs) -> None:
    """Save data to JSON file with proper serialization of numpy arrays"""
    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder, **kwargs)

def load_json(filepath: str) -> Any:
    """Load JSON file with proper handling of numeric types"""
    with open(filepath, 'r') as f:
        return json.load(f)

class DistributedMILExperimentRunner:
    def __init__(
        self, 
        data_root: str, 
        output_root: str,
        site_years: List[str],
        bag_durations: List[int],
        instance_durations: List[int],
        batch_size: int,
        skip_bag_creation: bool = False,
        force_bag_creation: bool = False,
        world_size: int = None
    ):
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        self.site_years = site_years
        self.bag_durations = bag_durations
        self.instance_durations = instance_durations
        self.batch_size = batch_size
        self.skip_bag_creation = skip_bag_creation
        self.force_bag_creation = force_bag_creation
        
        # Set up distributed training
        self.world_size = world_size or torch.cuda.device_count()
        if self.world_size == 0:
            self.world_size = 1
            logger.warning("No CUDA devices found, using CPU only.")
        
        # Add retry parameters
        self.max_retries = 3
        self.retry_delay = 2  # seconds
    
    def find_free_port(self) -> int:
        """Find a free port to use for distributed training"""
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
            return port
            
    def setup_distributed(self, rank: int) -> bool:
        """Initialize distributed process group with retry logic"""
        if self.world_size <= 1:
            return True
            
        retries = 0
        while retries < self.max_retries:
            try:
                # Find a free port
                port = self.find_free_port()
                
                # Set environment variables
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = str(port)
                
                # Initialize process group
                dist.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "gloo",
                    init_method=f'env://',
                    world_size=self.world_size,
                    rank=rank
                )
                
                # Set GPU for this process
                if torch.cuda.is_available():
                    torch.cuda.set_device(rank)
                
                # Wait for all processes
                dist.barrier()
                return True
                
            except Exception as e:
                retries += 1
                if retries < self.max_retries:
                    logger.warning(
                        f"Failed to initialize process group (attempt {retries}/{self.max_retries}): {e}"
                        f"\nRetrying in {self.retry_delay} seconds..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to initialize process group after {self.max_retries} attempts: {e}")
                    return False
    
    def cleanup_distributed(self):
        """Cleanup distributed process group with error handling"""
        try:
            if self.world_size > 1 and dist.is_initialized():
                dist.destroy_process_group()
        except Exception as e:
            logger.warning(f"Error during distributed cleanup: {e}")
    
    def create_experiment_configurations(self) -> List[Dict[str, Any]]:
        """Generate experiment configurations using provided parameters"""
        configurations = []
        for bag_duration in self.bag_durations:
            for instance_duration in self.instance_durations:
                config = get_default_config()
                config['training']['bag_duration'] = bag_duration
                config['training']['instance_duration'] = instance_duration
                config['training']['batch_size'] = self.batch_size
                config['paths']['results_dir'] = str(
                    self.output_root / f"results_bag{bag_duration}_inst{instance_duration}"
                )
                configurations.append(config)
        
        return configurations
    
    def run_experiment(self, rank: int, config: dict):
        """Run single experiment on one GPU"""
        setup_success = False
        try:
            # Setup distributed process
            setup_success = self.setup_distributed(rank)
            if not setup_success and self.world_size > 1:
                logger.error(f"Failed to setup distributed training on rank {rank}")
                return None, None
            
            # Unique experiment identifier
            exp_key = (
                f"bag{config['training']['bag_duration']}_"
                f"inst{config['training']['instance_duration']}"
            )
            
            if rank == 0:
                logger.info(f"Running experiment: {exp_key}")
            
            # Check if bags need to be created
            output_dir = Path(config['paths']['results_dir'])
            bags_exist = output_dir.exists() and any(output_dir.iterdir())
            
            create_bags = (
                self.force_bag_creation or 
                (not self.skip_bag_creation and not bags_exist)
            )
            
            # Synchronize bag creation decision across GPUs
            if self.world_size > 1:
                create_bags_tensor = torch.tensor(1 if create_bags else 0, device=rank if torch.cuda.is_available() else 'cpu')
                dist.broadcast(create_bags_tensor, src=0)
                create_bags = bool(create_bags_tensor.item())
            
            if create_bags:
                if rank == 0:
                    bag_creator = DynamicWhaleBagCreator(
                        root_dir=self.data_root,
                        output_dir=output_dir,
                        bag_duration=config['training']['bag_duration'],
                        instance_duration=config['training']['instance_duration']
                    )
                    
                    for site_year in self.site_years:
                        bag_creator.process_site(site_year)
                
                # Wait for bag creation to complete
                if self.world_size > 1:
                    dist.barrier()
            
            # Run distributed cross-validation
            results = self.run_distributed_cross_validation(
                config=config,
                rank=rank
            )
            
            # Save results (only on rank 0)
            if rank == 0:
                results_path = output_dir / 'experiment_results.json'
                save_json(results, str(results_path), indent=2)
            
            return exp_key, results

        except Exception as e:
            logger.error(f"Error in experiment on rank {rank}: {e}")
            return None, None    

        finally:
            if setup_success and self.world_size > 1:
                self.cleanup_distributed()
    
    def run_distributed_cross_validation(self, config: dict, rank: int):
        """Run cross-validation with distributed training"""
        # Import here to avoid circular imports
        from scripts.train import run_cross_validation
        
        # Modify config for distributed training
        config['training']['distributed'] = True
        config['training']['world_size'] = self.world_size
        config['training']['rank'] = rank
        
        # Run cross-validation
        return run_cross_validation(
            data_dir=config['paths']['results_dir'],
            site_years=self.site_years,
            config=config
        )
    
    def run_comprehensive_experiments(self):
        """Run all experiments with distributed training"""
        experiment_configs = self.create_experiment_configurations()
        all_results = {}
        
        for config in experiment_configs:
            try:
                # Launch distributed processes for each experiment
                if self.world_size > 1:
                    mp.spawn(
                        self.run_experiment,
                        args=(config,),
                        nprocs=self.world_size,
                        join=True
                    )
                else:
                    # Single process mode
                    self.run_experiment(0, config)
                
                # Load results from saved file
                exp_key = (
                    f"bag{config['training']['bag_duration']}_"
                    f"inst{config['training']['instance_duration']}"
                )
                
                results_path = Path(config['paths']['results_dir']) / 'experiment_results.json'
                if results_path.exists():
                    results = load_json(str(results_path))
                    
                    all_results[exp_key] = {
                        'configuration': config,
                        'results': results
                    }
                else:
                    logger.warning(f"Results file not found for {exp_key}: {results_path}")
                    
            except Exception as e:
                logger.error(f"Error running experiment with config {config}: {e}")
                continue
        
        # Analyze and save final results
        if all_results:
            performance_summary = self.analyze_configuration_performance(all_results)
            
            summary_path = self.output_root / 'performance_summary.json'
            save_json(performance_summary, str(summary_path), indent=2)
            
            # Print summary
            self.print_performance_summary(all_results, performance_summary)
        
        return all_results
    
    def analyze_configuration_performance(self, all_results: Dict) -> Dict:
        """Analyze performance across configurations"""
        performance_summary = {}
        
        for config_key, experiment_data in all_results.items():
            metrics = experiment_data['results']['aggregate']
            config = experiment_data['configuration']['training']
            
            performance_summary[config_key] = {
                'bag_duration': config['bag_duration'],
                'instance_duration': config['instance_duration'],
                'mean_accuracy': metrics['accuracy']['mean'],
                'std_accuracy': metrics['accuracy']['std'],
                'mean_f1': metrics['f1']['mean'],
                'std_f1': metrics['f1']['std']
            }
        
        return performance_summary
    
    def print_performance_summary(self, all_results: Dict, performance_summary: Dict):
        """Print summary of experiment performance"""
        logger.info("\nExperiment Performance Summary:")
        logger.info("=" * 50)
        
        for key, perf in performance_summary.items():
            logger.info(f"\n{key}:")
            logger.info(f"  Configuration: Bag Duration={perf['bag_duration']}s, "
                        f"Instance Duration={perf['instance_duration']}s")
            logger.info(f"  Mean F1: {perf['mean_f1']:.4f} ± {perf['std_f1']:.4f}")
            logger.info(f"  Mean Accuracy: {perf['mean_accuracy']:.4f} ± {perf['std_accuracy']:.4f}")
            
            # Print detailed fold results if available
            results = all_results[key]['results']
            if 'folds' in results:
                logger.info("\n  Per-Fold Results:")
                for i, fold in enumerate(results['folds']):
                    if 'metrics' in fold:
                        logger.info(f"    Fold {i+1} (Test: {fold['test_site']}): "
                                    f"F1={fold['metrics']['f1']:.4f}, "
                                    f"Acc={fold['metrics']['accuracy']:.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description='Run distributed MIL experiments')
    
    # Add arguments for bag durations, instance durations, and batch size
    parser.add_argument('--bag-durations', type=int, nargs='+', default=[300],
                      help='List of bag durations in seconds')
    parser.add_argument('--instance-durations', type=int, nargs='+', default=[15],
                      help='List of instance durations in seconds')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    
    # Add other configuration arguments
    parser.add_argument('--skip-bag-creation', action='store_true',
                      help='Skip bag creation step')
    parser.add_argument('--force-bag-creation', action='store_true',
                      help='Force bag creation even if bags exist')
    
    parser.add_argument('--data-root', type=str, required=True,
                   help='Root directory containing the dataset')
    parser.add_argument('--output-root', type=str, required=True,
                    help='Root directory for output files')
                    
    parser.add_argument('--site-years', type=str, nargs='+', required=True,
                       help='List of site years to process')
                       
    parser.add_argument('--single-process', action='store_true',
                      help='Run in single process mode (no distributed)')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Get number of available GPUs
    world_size = 1 if args.single_process else torch.cuda.device_count()
    if world_size == 0:
        world_size = 1
        logger.warning("No CUDA devices available, using CPU only.")
    
    logger.info(f"Running with {world_size} {'processes' if world_size > 1 else 'process'}")
    
    try:
        # Create distributed experiment runner
        runner = DistributedMILExperimentRunner(
            data_root=args.data_root,
            output_root=args.output_root,
            site_years=args.site_years,
            bag_durations=args.bag_durations,
            instance_durations=args.instance_durations,
            batch_size=args.batch_size,
            skip_bag_creation=args.skip_bag_creation,
            force_bag_creation=args.force_bag_creation,
            world_size=world_size
        )
        
        # Run experiments
        results = runner.run_comprehensive_experiments()
        
        if not results:
            logger.warning("No results were obtained. Check for errors.")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    import sys
    import numpy as np
    main()
