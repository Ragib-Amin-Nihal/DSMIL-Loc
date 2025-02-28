#!/usr/bin/env python3
"""
Script for training a whale call detection model using cross-validation
"""

import torch
import numpy as np
from pathlib import Path
import json
import logging
import argparse
import os
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

from data.dataset import LocalizationDataset
from data.collate import collate_whale_bags
from models.mil_net import ImprovedLocalizationMILNet
from training.losses import StableMILLoss
from training.earlystopping import TrendBasedEarlyStopping
from utils.visualization import ResultVisualizer
from utils.config_loader import ConfigLoader, get_default_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_epoch(model, loader, criterion, optimizer, device, config):
    """Run single training epoch with stability improvements"""
    epoch_metrics = {'loss': 0.0, 'f1': 0.0}
    num_batches = len(loader)
    model.train()
    
    for batch_idx, batch in enumerate(tqdm(loader, desc='Training')):
        if batch is None:
            continue
            
        # Move data to device
        spectrograms = batch['spectrograms'].to(device)
        features = batch['features'].to(device)
        labels = batch['bag_labels'].to(device)
        num_instances = batch['num_instances'].to(device)
        
        # Forward pass
        outputs = model(spectrograms, features, num_instances)
        loss = criterion(outputs, labels)
        
        # Backward pass with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config['training']['max_grad_norm']
        )
        
        optimizer.step()
        
        # Update metrics
        epoch_metrics['loss'] += loss.item() / num_batches
        
        # Compute F1 score
        predictions = (torch.sigmoid(outputs['logits']) > 0.5).float()
        tp = torch.sum((predictions == 1) & (labels == 1)).float()
        fp = torch.sum((predictions == 1) & (labels == 0)).float()
        fn = torch.sum((predictions == 0) & (labels == 1)).float()
        
        batch_f1 = tp / (tp + 0.5 * (fp + fn)) if tp + fp + fn > 0 else torch.tensor(0.0)
        epoch_metrics['f1'] += batch_f1.item() / num_batches
        
        # Log progress
        if (batch_idx + 1) % config['logging']['log_every'] == 0:
            logger.info(f"Batch {batch_idx+1}/{num_batches} - Loss: {loss.item():.4f}, F1: {batch_f1.item():.4f}")
    
    return epoch_metrics

def evaluate(model, loader, criterion, device):
    """Evaluate model on validation or test set"""
    model.eval()
    total_loss = 0
    predictions = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            if batch is None:
                continue
                
            # Move data to device
            spectrograms = batch['spectrograms'].to(device)
            features = batch['features'].to(device)
            batch_labels = batch['bag_labels'].to(device)
            num_instances = batch['num_instances'].to(device)
            
            # Forward pass
            outputs = model(spectrograms, features, num_instances)
            loss = criterion(outputs, batch_labels)
            
            # Store loss
            total_loss += loss.item()
            
            # Get predictions
            batch_preds = (torch.sigmoid(outputs['logits']) > 0.5).float()
            
            # Store predictions and labels
            predictions.extend(batch_preds.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Compute metrics
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    
    metrics = {
        'loss': total_loss / len(loader),
        'accuracy': (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0,
        'precision': tp / (tp + fp) if tp + fp > 0 else 0,
        'recall': tp / (tp + fn) if tp + fn > 0 else 0,
        'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_negatives': int(tn),
        'predictions': predictions,
        'labels': labels
    }
    
    return metrics

def train_model(model, train_loader, val_loader, device, config, output_dir):
    """Train model with early stopping and save checkpoints"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize early stopping
    early_stopping = TrendBasedEarlyStopping(
        patience=config['training']['patience'],
        window_size=10,
        min_epochs=config['training']['warmup_epochs'],
        min_improvement=0.01
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Initialize loss criterion
    criterion = StableMILLoss(
        pos_weight=torch.tensor([3.5]).to(device),
        smooth_factor=config['training']['label_smoothing'],
        focal_gamma=config['training']['focal_gamma']
    )
    
    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Training history
    history = {
        'train_f1': [],
        'val_f1': [],
        'train_loss': [],
        'val_loss': [],
        'learning_rates': [],
        'trends': []
    }
    
    # Define checkpoint path
    checkpoint_path = output_dir / 'best_model.pt'
    best_model_state = None
    
    logger.info("Starting training...")
    for epoch in range(config['training']['num_epochs']):
        # Training phase
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, config)
        
        # Validation phase
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_metrics['f1'])
        
        # Store history
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch summary
        logger.info(f"\nEpoch {epoch+1}:")
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}")
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # Check early stopping with trend analysis
        should_stop, trend_info = early_stopping(val_metrics['f1'], epoch)
        history['trends'].append(trend_info)
        
        # Save best model
        if val_metrics['f1'] > early_stopping.best_f1:
            logger.info(f"New best model! F1: {val_metrics['f1']:.4f}")
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }
            torch.save(best_model_state, checkpoint_path)
            
            # Print detailed metrics for best model
            print_metrics(val_metrics)
        
        # Check for early stopping
        if should_stop:
            logger.info(f"\nEarly stopping triggered:")
            logger.info(f"Status: {trend_info['status']}")
            logger.info(f"Best F1: {trend_info['best_f1']:.4f} at epoch {trend_info['best_epoch']}")
            break
    
    # Save final training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x)
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state_dict'])
    
    return model, history, checkpoint_path

def print_metrics(metrics):
    """Print detailed metrics"""
    logger.info("\nDetailed Metrics:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    logger.info(f"True Positives: {metrics['true_positives']}")
    logger.info(f"False Positives: {metrics['false_positives']}")
    logger.info(f"True Negatives: {metrics['true_negatives']}")
    logger.info(f"False Negatives: {metrics['false_negatives']}")
    logger.info(f"Loss: {metrics['loss']:.4f}")

def create_cv_splits(site_years):
    """Create blocked cross-validation splits"""
    cv_splits = []
    for test_site in site_years:
        train_sites = [site for site in site_years if site != test_site]
        cv_splits.append((train_sites, test_site))
    return cv_splits

def run_cross_validation(data_dir, site_years, config):
    """Run cross-validation with visualization"""
    device = torch.device(config['training']['device'])
    results_dir = Path(config['paths']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create splits
    cv_splits = create_cv_splits(site_years)
    cv_results = []
    
    logger.info(f"Starting {len(cv_splits)}-fold cross-validation")
    
    for fold, (train_sites, test_site) in enumerate(cv_splits, 1):
        # Initialize wandb run if enabled
        if config['wandb']['enabled'] and WANDB_AVAILABLE:
            import wandb
            wandb.init(
                project=config['wandb']['project'],
                name=f"fold_{fold}_{test_site}",
                config=config
            )
        
        fold_dir = results_dir / f'fold_{fold}'
        fold_dir.mkdir(exist_ok=True)
        
        # Setup data
        val_site = train_sites[-1]
        train_sites = train_sites[:-1]
        
        logger.info(f"Fold {fold}: Training on {train_sites}, validating on {val_site}, testing on {test_site}")
        
        # Create datasets
        train_dataset = LocalizationDataset(data_dir, train_sites, preload_data=False)
        val_dataset = LocalizationDataset(data_dir, [val_site], preload_data=False)
        test_dataset = LocalizationDataset(data_dir, [test_site], preload_data=False)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            collate_fn=collate_whale_bags
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            collate_fn=collate_whale_bags
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            collate_fn=collate_whale_bags
        )
        
        # Initialize model
        model = ImprovedLocalizationMILNet(
            feature_dim=config['model']['feature_dim'],
            num_heads=config['model']['num_heads']
        ).to(device)
        
        # Train model
        trained_model, history, checkpoint_path = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config,
            output_dir=fold_dir
        )
        
        # Initialize loss criterion for evaluation
        criterion = StableMILLoss(
            pos_weight=torch.tensor([3.5]).to(device),
            smooth_factor=config['training']['label_smoothing'],
            focal_gamma=config['training']['focal_gamma']
        )
        
        # Evaluate on test set
        test_metrics = evaluate(trained_model, test_loader, criterion, device)
        
        # Visualize results
        visualizer = ResultVisualizer(save_dir=fold_dir)
        visualizer.plot_confusion_matrix(test_metrics)
        
        # Store fold results
        fold_results = {
            'fold': fold,
            'test_site': test_site,
            'val_site': val_site,
            'train_sites': train_sites,
            'metrics': test_metrics,
            'history': history
        }
        
        cv_results.append(fold_results)
        
        # Save fold results
        with open(fold_dir / 'fold_results.json', 'w') as f:
            json.dump(fold_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x)
        
        # Log to wandb if enabled
        if config['wandb']['enabled'] and WANDB_AVAILABLE:
            wandb.log({"test_metrics": test_metrics})
            wandb.finish()
    
    # Compute aggregate results
    aggregate_metrics = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        values = [fold['metrics'][metric] for fold in cv_results]
        aggregate_metrics[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
    
    # Save final results
    final_results = {
        'folds': cv_results,
        'aggregate': aggregate_metrics
    }
    
    with open(results_dir / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x)
    
    # Print final results
    logger.info("\nFinal Cross-Validation Results:")
    for metric, stats in aggregate_metrics.items():
        logger.info(f"{metric.capitalize()}:")
        logger.info(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
        logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    return final_results

def parse_args():
    parser = argparse.ArgumentParser(description='Train a whale call detection model with cross-validation')
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing processed data')
    
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save results')
    
    parser.add_argument('--site-years', type=str, nargs='+',
                       help='List of site years to use for cross-validation (default: all available)')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    # Check if wandb is available
    global WANDB_AVAILABLE
    try:
        import wandb
        WANDB_AVAILABLE = True
    except ImportError:
        WANDB_AVAILABLE = False
        logger.warning("wandb not installed. Experiment tracking will be disabled.")
    
    # Load configuration
    try:
        config = ConfigLoader.load_config(args.config)
    except FileNotFoundError:
        logger.warning(f"Configuration file {args.config} not found. Using default configuration.")
        config = get_default_config()
    
    # Update config with command line arguments
    config['paths']['data_root'] = args.data_dir
    config['paths']['results_dir'] = args.output_dir
    config['training']['seed'] = args.seed
    
    # Determine site years
    if args.site_years:
        site_years = args.site_years
    else:
        # Get all site years from data directory
        data_dir = Path(args.data_dir)
        site_years = [d.name for d in data_dir.iterdir() if d.is_dir()]
    
    logger.info(f"Using site years: {site_years}")
    
    # Run cross-validation
    try:
        results = run_cross_validation(args.data_dir, site_years, config)
        logger.info("Training completed successfully.")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()
