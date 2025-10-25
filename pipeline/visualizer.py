"""
Visualization module for the SAPPHIRE pipeline.
Creates comprehensive visualizations for feature analysis and model results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
warnings.filterwarnings('ignore')

# Advanced plotting libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches

from .config import config

logger = logging.getLogger(__name__)

class Visualizer:
    """
    Comprehensive visualization suite for music analysis results.
    """
    
    def __init__(self, config_obj=None):
        self.config = config_obj or config
        self.logger = logging.getLogger(__name__)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Color schemes for different mood clusters
        self.mood_colors = {
            'Cluster 1': '#FF6B6B',  # Red - Boisterous/Passionate
            'Cluster 2': '#4ECDC4',  # Teal - Cheerful/Sweet
            'Cluster 3': '#45B7D1',  # Blue - Autumnal/Poignant
            'Cluster 4': '#96CEB4',  # Green - Humorous/Whimsical
            'Cluster 5': '#FFEAA7'   # Yellow - Aggressive/Intense
        }
    
    def create_feature_distribution_plots(self, features_df: pd.DataFrame, output_dir: str):
        """
        Create distribution plots for all features.
        
        Args:
            features_df: DataFrame with features
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir) / "feature_distributions"
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Creating feature distribution plots...")
        
        # Get numeric columns
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        
        # Create distribution plots in batches
        batch_size = 16
        for i in range(0, len(numeric_cols), batch_size):
            batch_cols = numeric_cols[i:i+batch_size]
            
            fig, axes = plt.subplots(4, 4, figsize=(20, 16))
            axes = axes.flatten()
            
            for j, col in enumerate(batch_cols):
                if j < len(axes):
                    ax = axes[j]
                    
                    # Handle infinite and NaN values
                    data = features_df[col].replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if len(data) > 0:
                        ax.hist(data, bins=30, alpha=0.7, edgecolor='black')
                        ax.set_title(f'{col}', fontsize=10)
                        ax.set_xlabel('Value')
                        ax.set_ylabel('Frequency')
                    else:
                        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'{col} (No Data)', fontsize=10)
            
            # Hide unused subplots
            for j in range(len(batch_cols), len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(output_path / f'distributions_batch_{i//batch_size + 1}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"Feature distribution plots saved to {output_path}")
    
    def create_mood_analysis_plots(self, features_df: pd.DataFrame, output_dir: str):
        """
        Create mood-specific analysis plots.
        
        Args:
            features_df: DataFrame with features and mood labels
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir) / "mood_analysis"
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Creating mood analysis plots...")
        
        if 'mood_cluster' not in features_df.columns:
            self.logger.warning("No mood_cluster column found, skipping mood analysis plots")
            return
        
        # Mood distribution
        plt.figure(figsize=(10, 6))
        mood_counts = features_df['mood_cluster'].value_counts()
        colors = [self.mood_colors.get(mood, '#95A5A6') for mood in mood_counts.index]
        
        bars = plt.bar(mood_counts.index, mood_counts.values, color=colors, alpha=0.8, edgecolor='black')
        plt.title('Distribution of Mood Clusters', fontsize=16, fontweight='bold')
        plt.xlabel('Mood Cluster')
        plt.ylabel('Number of Tracks')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / 'mood_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature comparison across moods
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        important_features = [col for col in numeric_cols if any(
            keyword in col.lower() for keyword in ['mfcc', 'spectral', 'tempo', 'sentiment', 'energy']
        )][:12]  # Top 12 important features
        
        if important_features:
            fig, axes = plt.subplots(3, 4, figsize=(20, 15))
            axes = axes.flatten()
            
            for i, feature in enumerate(important_features):
                ax = axes[i]
                
                # Create box plot for each mood
                mood_data = []
                mood_labels = []
                
                for mood in features_df['mood_cluster'].unique():
                    if pd.notna(mood):
                        data = features_df[features_df['mood_cluster'] == mood][feature]
                        data = data.replace([np.inf, -np.inf], np.nan).dropna()
                        if len(data) > 0:
                            mood_data.append(data)
                            mood_labels.append(mood)
                
                if mood_data:
                    bp = ax.boxplot(mood_data, labels=mood_labels, patch_artist=True)
                    
                    # Color the boxes
                    for patch, mood in zip(bp['boxes'], mood_labels):
                        patch.set_facecolor(self.mood_colors.get(mood, '#95A5A6'))
                        patch.set_alpha(0.7)
                
                ax.set_title(feature, fontsize=10)
                ax.tick_params(axis='x', rotation=45)
            
            # Hide unused subplots
            for i in range(len(important_features), len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle('Feature Comparison Across Mood Clusters', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path / 'mood_feature_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_correlation_heatmap(self, features_df: pd.DataFrame, output_dir: str, max_features: int = 50):
        """
        Create correlation heatmap for features.
        
        Args:
            features_df: DataFrame with features
            output_dir: Directory to save plots
            max_features: Maximum number of features to include
        """
        output_path = Path(output_dir) / "correlations"
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Creating correlation heatmap...")
        
        # Get numeric columns
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        
        # Select most important features
        if len(numeric_cols) > max_features:
            # Calculate variance to select most informative features
            variances = features_df[numeric_cols].var().sort_values(ascending=False)
            selected_cols = variances.head(max_features).index
        else:
            selected_cols = numeric_cols
        
        # Calculate correlation matrix
        corr_data = features_df[selected_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        correlation_matrix = corr_data.corr()
        
        # Create heatmap
        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save correlation matrix as CSV
        correlation_matrix.to_csv(output_path / 'correlation_matrix.csv')
        
        self.logger.info(f"Correlation analysis saved to {output_path}")
    
    def create_pca_visualization(self, features_df: pd.DataFrame, output_dir: str):
        """
        Create PCA visualization of features.
        
        Args:
            features_df: DataFrame with features
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir) / "dimensionality_reduction"
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Creating PCA visualization...")
        
        # Prepare data
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        X = features_df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Explained variance plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, min(21, len(pca.explained_variance_ratio_) + 1)), 
                pca.explained_variance_ratio_[:20], 'bo-')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Explained Variance')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        plt.plot(range(1, min(21, len(cumsum) + 1)), cumsum[:20], 'ro-')
        plt.xlabel('Principal Component')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'pca_explained_variance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2D PCA plot with mood coloring
        if 'mood_cluster' in features_df.columns:
            plt.figure(figsize=(12, 8))
            
            for mood in features_df['mood_cluster'].unique():
                if pd.notna(mood):
                    mask = features_df['mood_cluster'] == mood
                    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                              c=self.mood_colors.get(mood, '#95A5A6'),
                              label=mood, alpha=0.7, s=50)
            
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.title('PCA Visualization by Mood Cluster')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / 'pca_mood_clusters.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"PCA visualization saved to {output_path}")
    
    def create_tsne_visualization(self, features_df: pd.DataFrame, output_dir: str):
        """
        Create t-SNE visualization of features.
        
        Args:
            features_df: DataFrame with features
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir) / "dimensionality_reduction"
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Creating t-SNE visualization...")
        
        # Prepare data
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        X = features_df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform t-SNE (limit samples for performance)
        max_samples = 1000
        if len(X_scaled) > max_samples:
            indices = np.random.choice(len(X_scaled), max_samples, replace=False)
            X_tsne_input = X_scaled[indices]
            features_subset = features_df.iloc[indices]
        else:
            X_tsne_input = X_scaled
            features_subset = features_df
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_tsne_input)-1))
        X_tsne = tsne.fit_transform(X_tsne_input)
        
        # Create t-SNE plot with mood coloring
        if 'mood_cluster' in features_subset.columns:
            plt.figure(figsize=(12, 8))
            
            for mood in features_subset['mood_cluster'].unique():
                if pd.notna(mood):
                    mask = features_subset['mood_cluster'] == mood
                    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                              c=self.mood_colors.get(mood, '#95A5A6'),
                              label=mood, alpha=0.7, s=50)
            
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.title('t-SNE Visualization by Mood Cluster')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / 'tsne_mood_clusters.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"t-SNE visualization saved to {output_path}")
    
    def create_cross_modal_plots(self, cross_modal_results: Dict[str, Any], output_dir: str):
        """
        Create visualizations for cross-modal analysis results.
        
        Args:
            cross_modal_results: Results from cross-modal analysis
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir) / "cross_modal"
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Creating cross-modal analysis plots...")
        
        # Cross-correlation heatmap
        if 'correlations' in cross_modal_results and cross_modal_results['correlations']:
            corr_data = cross_modal_results['correlations']
            
            plt.figure(figsize=(14, 10))
            sns.heatmap(corr_data['matrix'], 
                       xticklabels=corr_data['lyrical_features'],
                       yticklabels=corr_data['acoustic_features'],
                       cmap='coolwarm', center=0, annot=False)
            
            plt.title('Cross-Modal Feature Correlations\n(Acoustic vs Lyrical Features)', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Lyrical Features')
            plt.ylabel('Acoustic Features')
            plt.tight_layout()
            plt.savefig(output_path / 'cross_modal_correlations.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Jaccard similarity distribution
        if 'jaccard_similarity' in cross_modal_results and cross_modal_results['jaccard_similarity']:
            jaccard_data = cross_modal_results['jaccard_similarity']
            
            plt.figure(figsize=(10, 6))
            plt.hist(jaccard_data['scores'], bins=30, alpha=0.7, edgecolor='black', color='skyblue')
            plt.axvline(jaccard_data['mean'], color='red', linestyle='--', 
                       label=f'Mean: {jaccard_data["mean"]:.3f}')
            plt.xlabel('Jaccard Similarity Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of Cross-Modal Jaccard Similarity Scores')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / 'jaccard_similarity_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"Cross-modal plots saved to {output_path}")
    
    def create_model_performance_plots(self, model_results: Dict[str, Dict], output_dir: str):
        """
        Create model performance visualization plots.
        
        Args:
            model_results: Results from model training
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir) / "model_performance"
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Creating model performance plots...")
        
        # Model comparison bar plot
        models = list(model_results.keys())
        test_accuracies = [model_results[m]['test_accuracy'] for m in models]
        cv_means = [model_results[m]['cv_mean'] for m in models]
        cv_stds = [model_results[m]['cv_std'] for m in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Test accuracy comparison
        bars1 = ax1.bar(models, test_accuracies, color='lightblue', alpha=0.8, edgecolor='black')
        ax1.set_title('Model Test Accuracy Comparison')
        ax1.set_ylabel('Test Accuracy')
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bar, acc in zip(bars1, test_accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Cross-validation comparison with error bars
        bars2 = ax2.bar(models, cv_means, yerr=cv_stds, capsize=5, 
                        color='lightcoral', alpha=0.8, edgecolor='black')
        ax2.set_title('Model Cross-Validation Performance')
        ax2.set_ylabel('CV Accuracy')
        ax2.set_ylim(0, 1)
        
        # Add value labels
        for bar, mean, std in zip(bars2, cv_means, cv_stds):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.01,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Rotate x-axis labels
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Model performance plots saved to {output_path}")
    
    def create_feature_importance_plot(self, importance_df: pd.DataFrame, output_dir: str, top_n: int = 20):
        """
        Create feature importance visualization.
        
        Args:
            importance_df: DataFrame with feature importance
            output_dir: Directory to save plots
            top_n: Number of top features to show
        """
        output_path = Path(output_dir) / "feature_importance"
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Creating feature importance plot...")
        
        # Select top features
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                       color='steelblue', alpha=0.8, edgecolor='black')
        
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{importance:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(output_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save importance data
        importance_df.to_csv(output_path / 'feature_importance.csv', index=False)
        
        self.logger.info(f"Feature importance plot saved to {output_path}")
    
    def create_comprehensive_report(self, features_df: pd.DataFrame, analysis_results: Dict[str, Any], 
                                  model_results: Dict[str, Dict], output_dir: str):
        """
        Create a comprehensive visualization report.
        
        Args:
            features_df: DataFrame with features
            analysis_results: Results from analysis
            model_results: Results from model training
            output_dir: Directory to save report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Creating comprehensive visualization report...")
        
        # Create all visualizations
        self.create_feature_distribution_plots(features_df, str(output_path))
        self.create_mood_analysis_plots(features_df, str(output_path))
        self.create_correlation_heatmap(features_df, str(output_path))
        self.create_pca_visualization(features_df, str(output_path))
        self.create_tsne_visualization(features_df, str(output_path))
        
        if 'cross_modal_analysis' in analysis_results:
            self.create_cross_modal_plots(analysis_results['cross_modal_analysis'], str(output_path))
        
        if model_results:
            self.create_model_performance_plots(model_results, str(output_path))
        
        # Create summary statistics
        summary_stats = {
            'total_tracks': len(features_df),
            'total_features': len(features_df.select_dtypes(include=[np.number]).columns),
            'mood_distribution': features_df['mood_cluster'].value_counts().to_dict() if 'mood_cluster' in features_df.columns else {},
            'dataset_distribution': features_df['dataset'].value_counts().to_dict() if 'dataset' in features_df.columns else {}
        }
        
        # Save summary
        import json
        with open(output_path / 'summary_statistics.json', 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        self.logger.info(f"Comprehensive visualization report created in {output_path}")