import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)

class ResultVisualizer:
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def plot_attention(self, model, dataset, num_examples: int = 5):
        """Visualize attention weights with improved error handling"""
        try:
            model.eval()
            device = next(model.parameters()).device
            
            # Create figure
            fig, axes = plt.subplots(min(num_examples, len(dataset)), 2, 
                                   figsize=(15, 5*min(num_examples, len(dataset))))
            
            # Handle single example case
            if num_examples == 1:
                axes = axes.reshape(1, -1)
            
            with torch.no_grad():
                for i in range(min(num_examples, len(dataset))):
                    try:
                        data = dataset[i]
                        
                        # Move data to device
                        spectrograms = data['spectrograms'].unsqueeze(0).to(device)
                        features = data['features'].unsqueeze(0).to(device)
                        num_instances = data['num_instances'].unsqueeze(0).to(device)
                        
                        # Get model outputs
                        outputs = model(spectrograms, features, num_instances)
                        attention = outputs['attention_weights'].squeeze().cpu().numpy()
                        
                        # Plot spectrogram
                        axes[i, 0].imshow(spectrograms.cpu()[0, 0].T, 
                                        aspect='auto', origin='lower')
                        axes[i, 0].set_title(f"Spectrogram (Bag {i})")
                        
                        # Plot attention weights
                        axes[i, 1].bar(range(len(attention)), attention)
                        axes[i, 1].set_title("Attention Weights")
                        axes[i, 1].set_xlabel("Instance")
                        axes[i, 1].set_ylabel("Weight")
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing example {i}: {str(e)}")
                        continue
            
            plt.tight_layout()
            
            # Save figure
            save_path = self.save_dir / 'attention_visualization.png'
            plt.savefig(save_path)
            plt.close()
            
            self.logger.info(f"Attention visualization saved to {save_path}")
            
        except Exception as e:
            self.logger.warning(f"Error plotting attention weights: {str(e)}")
        
    def plot_embeddings(self, metrics: dict):
        """Visualize instance embeddings with improved error handling"""
        try:
            # Check for required data
            if 'instance_embeddings' not in metrics:
                self.logger.warning("Missing instance embeddings for visualization")
                return
                
            if 'predictions' not in metrics:
                self.logger.warning("Missing predictions for visualization")
                return
            
            # Convert to numpy arrays
            embeddings = np.array(metrics['instance_embeddings'])
            predictions = np.array(metrics['predictions'])
            
            self.logger.info(f"Embeddings shape: {embeddings.shape}")
            self.logger.info(f"Predictions shape: {predictions.shape}")
            
            # Create visualization
            plt.figure(figsize=(10, 6))
            
            if embeddings.ndim == 2 and embeddings.shape[0] >= 2:
                # Use PCA for dimensionality reduction
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                embeddings_2d = pca.fit_transform(embeddings)
                
                # Adjust predictions if needed
                if len(predictions) != len(embeddings_2d):
                    predictions = predictions[:len(embeddings_2d)]
                
                scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                   c=predictions, cmap='coolwarm', alpha=0.6)
                plt.colorbar(scatter, label='Prediction')
                plt.title("PCA Visualization of Instance Embeddings")
                
            else:
                plt.text(0.5, 0.5, "Insufficient data for visualization",
                        ha='center', va='center')
                plt.title("Embedding Visualization Not Possible")
            
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            
            # Save figure
            save_path = self.save_dir / 'embedding_visualization.png'
            plt.savefig(save_path)
            plt.close()
            
            self.logger.info(f"Embeddings visualization saved to {save_path}")
            
        except Exception as e:
            self.logger.warning(f"Error plotting embeddings: {str(e)}")

    def plot_confusion_matrix(self, metrics: dict):
        """Visualize confusion matrix with improved error handling"""
        try:
            # Check if confusion matrix exists in metrics
            if 'confusion_matrix' not in metrics:
                # Try to compute confusion matrix from predictions and labels
                if 'predictions' in metrics and 'labels' in metrics:
                    from sklearn.metrics import confusion_matrix
                    predictions = np.array(metrics['predictions'])
                    labels = np.array(metrics['labels'])
                    cm = confusion_matrix(labels, predictions)
                else:
                    self.logger.warning("Cannot create confusion matrix: missing required data")
                    return
            else:
                cm = np.array(metrics['confusion_matrix'])
            
            # Create figure
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            
            # Add text annotations
            thresh = cm.max() / 2
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            # Add labels
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Add value labels
            plt.colorbar(label='Count')
            
            # Save figure
            save_path = self.save_dir / 'confusion_matrix.png'
            plt.savefig(save_path)
            plt.close()
            
            self.logger.info(f"Confusion matrix saved to {save_path}")
            
        except Exception as e:
            self.logger.warning(f"Error plotting confusion matrix: {str(e)}")
