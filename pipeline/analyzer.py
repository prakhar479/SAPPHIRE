"""
Comprehensive analysis module for the SAPPHIRE pipeline.
Handles statistical analysis, clustering, and feature importance analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Statistical and ML libraries
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, pearsonr

from .config import config

logger = logging.getLogger(__name__)


class Analyzer:
    """
    Comprehensive analyzer for music feature analysis and clustering.
    Implements the analytical methodology described in SAPPHIRE Sprint 3.
    """

    def __init__(self, config_obj=None):
        self.config = config_obj or config
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()

    def analyze_dataset(
        self, features_df: pd.DataFrame, output_dir: str
    ) -> Dict[str, Any]:
        """
        Perform comprehensive dataset analysis.

        Args:
            features_df: DataFrame with extracted features
            output_dir: Directory to save analysis results

        Returns:
            Dictionary with analysis results
        """
        self.logger.info("Starting comprehensive dataset analysis...")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {
            "timestamp": datetime.now().isoformat(),
            "dataset_stats": {},
            "feature_analysis": {},
            "clustering_analysis": {},
            "correlation_analysis": {},
            "dimensionality_analysis": {},
        }

        # Basic dataset statistics
        results["dataset_stats"] = self._get_dataset_statistics(features_df)

        # Feature analysis
        results["feature_analysis"] = self._analyze_features(features_df, output_path)

        # Correlation analysis
        results["correlation_analysis"] = self._analyze_correlations(
            features_df, output_path
        )

        # Dimensionality analysis (PCA)
        results["dimensionality_analysis"] = self._analyze_dimensionality(
            features_df, output_path
        )

        # Clustering analysis
        results["clustering_analysis"] = self._analyze_clustering(
            features_df, output_path
        )

        # Save comprehensive report
        self._save_analysis_report(results, output_path / "analysis_report.json")

        self.logger.info(f"Analysis complete. Results saved to {output_path}")
        return results

    def _get_dataset_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset statistics."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        stats = {
            "total_tracks": len(df),
            "total_features": len(numeric_cols),
            "missing_values": df[numeric_cols].isnull().sum().sum(),
            "feature_types": {
                "acoustic": len([c for c in numeric_cols if "acoustic" in c]),
                "rhythm": len([c for c in numeric_cols if "rhythm" in c]),
                "harmony": len([c for c in numeric_cols if "harmony" in c]),
                "lyrics": len([c for c in numeric_cols if "lyrics" in c]),
                "quality": len([c for c in numeric_cols if "quality" in c]),
            },
        }

        # Dataset distribution by source
        if "meta_dataset" in df.columns:
            stats["dataset_distribution"] = df["meta_dataset"].value_counts().to_dict()

        return stats

    def _analyze_features(self, df: pd.DataFrame, output_path: Path) -> Dict[str, Any]:
        """Analyze individual features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Feature statistics
        feature_stats = df[numeric_cols].describe().transpose()

        # Feature variability (coefficient of variation)
        cv = df[numeric_cols].std() / df[numeric_cols].mean()
        cv = cv.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Feature distributions
        self._plot_feature_distributions(df[numeric_cols], output_path)

        return {
            "feature_statistics": feature_stats.to_dict(),
            "coefficient_of_variation": cv.to_dict(),
            "high_variance_features": cv.nlargest(10).to_dict(),
            "low_variance_features": cv.nsmallest(10).to_dict(),
        }

    def _analyze_correlations(
        self, df: pd.DataFrame, output_path: Path
    ) -> Dict[str, Any]:
        """Analyze feature correlations."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Compute correlation matrices
        pearson_corr = df[numeric_cols].corr(method="pearson")
        spearman_corr = df[numeric_cols].corr(method="spearman")

        # Find highly correlated features
        high_corr_pairs = self._find_high_correlations(pearson_corr, threshold=0.8)

        # Cross-modal correlations (if we have different feature types)
        cross_modal = self._analyze_cross_modal_correlations(df, numeric_cols)

        # Create correlation heatmaps
        self._plot_correlation_heatmaps(pearson_corr, spearman_corr, output_path)

        return {
            "high_correlation_pairs": high_corr_pairs,
            "cross_modal_correlations": cross_modal,
            "correlation_summary": {
                "mean_abs_correlation": float(np.abs(pearson_corr.values).mean()),
                "max_correlation": float(np.abs(pearson_corr.values).max()),
                "highly_correlated_pairs": len(high_corr_pairs),
            },
        }

    def _analyze_dimensionality(
        self, df: pd.DataFrame, output_path: Path
    ) -> Dict[str, Any]:
        """Analyze dimensionality using PCA."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_cols].fillna(0)

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # PCA analysis
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)

        # Explained variance analysis
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        # Find number of components for 80% and 95% variance
        n_components_80 = np.argmax(cumulative_variance >= 0.8) + 1
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

        # Component loadings
        loadings = pd.DataFrame(
            pca.components_[:10].T,  # First 10 components
            columns=[f"PC{i+1}" for i in range(10)],
            index=numeric_cols,
        )

        # Plot PCA results
        self._plot_pca_analysis(
            explained_variance_ratio, cumulative_variance, X_pca, loadings, output_path
        )

        return {
            "explained_variance_ratio": explained_variance_ratio[:20].tolist(),
            "cumulative_variance": cumulative_variance[:20].tolist(),
            "n_components_80_percent": int(n_components_80),
            "n_components_95_percent": int(n_components_95),
            "top_loadings": {
                f"PC{i+1}": loadings[f"PC{i+1}"].abs().nlargest(5).to_dict()
                for i in range(5)
            },
        }

    def perform_clustering(self, df: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
        """Public wrapper to perform clustering analysis only.

        This exposes the clustering functionality used internally by
        `analyze_dataset` so that other components (CLI, Pipeline) can
        request clustering without rerunning the full analysis.

        Args:
            df: Feature DataFrame.
            output_dir: Directory where clustering plots/results should be saved.

        Returns:
            Dictionary with clustering analysis results.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        return self._analyze_clustering(df, output_path)

    def _analyze_clustering(
        self, df: pd.DataFrame, output_path: Path
    ) -> Dict[str, Any]:
        """Perform clustering analysis."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)

        clustering_results = {}

        # K-means clustering with different k values
        kmeans_results = self._analyze_kmeans_clustering(X_scaled, output_path)
        clustering_results["kmeans"] = kmeans_results

        # Hierarchical clustering
        hierarchical_results = self._analyze_hierarchical_clustering(
            X_scaled, output_path
        )
        clustering_results["hierarchical"] = hierarchical_results

        # DBSCAN clustering
        dbscan_results = self._analyze_dbscan_clustering(X_scaled, output_path)
        clustering_results["dbscan"] = dbscan_results

        # If we have mood labels, compare with ground truth
        if "mood_cluster" in df.columns or "mood_category" in df.columns:
            ground_truth_analysis = self._analyze_ground_truth_clustering(df, X_scaled)
            clustering_results["ground_truth_comparison"] = ground_truth_analysis

        return clustering_results

    def _find_high_correlations(
        self, corr_matrix: pd.DataFrame, threshold: float = 0.8
    ) -> List[Dict]:
        """Find pairs of features with high correlation."""
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    high_corr_pairs.append(
                        {
                            "feature1": corr_matrix.columns[i],
                            "feature2": corr_matrix.columns[j],
                            "correlation": float(corr_val),
                        }
                    )

        return sorted(
            high_corr_pairs, key=lambda x: abs(x["correlation"]), reverse=True
        )

    def _analyze_cross_modal_correlations(
        self, df: pd.DataFrame, numeric_cols: List[str]
    ) -> Dict[str, Any]:
        """Analyze correlations between different feature modalities."""
        # Group features by modality
        modalities = {
            "acoustic": [c for c in numeric_cols if "acoustic" in c],
            "rhythm": [c for c in numeric_cols if "rhythm" in c],
            "harmony": [c for c in numeric_cols if "harmony" in c],
            "lyrics": [c for c in numeric_cols if "lyrics" in c],
            "quality": [c for c in numeric_cols if "quality" in c],
        }

        cross_modal_corr = {}

        for mod1, features1 in modalities.items():
            for mod2, features2 in modalities.items():
                if mod1 != mod2 and features1 and features2:
                    # Compute average correlation between modalities
                    corr_values = []
                    for f1 in features1:
                        for f2 in features2:
                            if f1 in df.columns and f2 in df.columns:
                                corr = df[f1].corr(df[f2])
                                if not np.isnan(corr):
                                    corr_values.append(abs(corr))

                    if corr_values:
                        cross_modal_corr[f"{mod1}_vs_{mod2}"] = {
                            "mean_abs_correlation": float(np.mean(corr_values)),
                            "max_abs_correlation": float(np.max(corr_values)),
                            "feature_pairs": len(corr_values),
                        }

        return cross_modal_corr

    def _analyze_kmeans_clustering(
        self, X_scaled: np.ndarray, output_path: Path
    ) -> Dict[str, Any]:
        """Analyze K-means clustering with different k values."""
        k_range = range(2, 11)
        silhouette_scores = []
        inertias = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)

            silhouette_avg = silhouette_score(X_scaled, labels)
            silhouette_scores.append(silhouette_avg)
            inertias.append(kmeans.inertia_)

        # Find optimal k using elbow method and silhouette score
        optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]

        # Perform clustering with optimal k
        kmeans_optimal = KMeans(
            n_clusters=optimal_k_silhouette, random_state=42, n_init=10
        )
        optimal_labels = kmeans_optimal.fit_predict(X_scaled)

        # Plot clustering results
        self._plot_clustering_analysis(X_scaled, optimal_labels, "K-means", output_path)

        return {
            "k_range": list(k_range),
            "silhouette_scores": silhouette_scores,
            "inertias": inertias,
            "optimal_k": int(optimal_k_silhouette),
            "optimal_silhouette_score": float(max(silhouette_scores)),
            "cluster_sizes": np.bincount(optimal_labels).tolist(),
        }

    def _analyze_hierarchical_clustering(
        self, X_scaled: np.ndarray, output_path: Path
    ) -> Dict[str, Any]:
        """Analyze hierarchical clustering."""
        # Try different numbers of clusters
        n_clusters_range = range(2, 11)
        silhouette_scores = []

        for n_clusters in n_clusters_range:
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
            labels = hierarchical.fit_predict(X_scaled)
            silhouette_avg = silhouette_score(X_scaled, labels)
            silhouette_scores.append(silhouette_avg)

        optimal_n = n_clusters_range[np.argmax(silhouette_scores)]

        # Perform optimal clustering
        hierarchical_optimal = AgglomerativeClustering(n_clusters=optimal_n)
        optimal_labels = hierarchical_optimal.fit_predict(X_scaled)

        return {
            "n_clusters_range": list(n_clusters_range),
            "silhouette_scores": silhouette_scores,
            "optimal_n_clusters": int(optimal_n),
            "optimal_silhouette_score": float(max(silhouette_scores)),
            "cluster_sizes": np.bincount(optimal_labels).tolist(),
        }

    def _analyze_dbscan_clustering(
        self, X_scaled: np.ndarray, output_path: Path
    ) -> Dict[str, Any]:
        """Analyze DBSCAN clustering."""
        # Try different eps values
        eps_range = np.arange(0.1, 2.0, 0.1)
        results = []

        for eps in eps_range:
            dbscan = DBSCAN(eps=eps, min_samples=5)
            labels = dbscan.fit_predict(X_scaled)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            if n_clusters > 1:
                # Only calculate silhouette score if we have more than 1 cluster
                silhouette_avg = silhouette_score(X_scaled, labels)
            else:
                silhouette_avg = -1

            results.append(
                {
                    "eps": float(eps),
                    "n_clusters": n_clusters,
                    "n_noise": n_noise,
                    "silhouette_score": float(silhouette_avg),
                }
            )

        # Find best eps
        valid_results = [r for r in results if r["silhouette_score"] > -1]
        if valid_results:
            best_result = max(valid_results, key=lambda x: x["silhouette_score"])
            return {
                "eps_range": [r["eps"] for r in results],
                "results": results,
                "best_eps": best_result["eps"],
                "best_n_clusters": best_result["n_clusters"],
                "best_silhouette_score": best_result["silhouette_score"],
            }
        else:
            return {
                "eps_range": [r["eps"] for r in results],
                "results": results,
                "note": "No valid clustering found",
            }

    def _analyze_ground_truth_clustering(
        self, df: pd.DataFrame, X_scaled: np.ndarray
    ) -> Dict[str, Any]:
        """Compare clustering results with ground truth mood labels."""
        results = {}

        # Use mood cluster labels if available
        if "mood_cluster" in df.columns:
            mood_labels = df["mood_cluster"].fillna("unknown")
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(mood_labels)

            # Compare with K-means clustering
            kmeans = KMeans(n_clusters=len(np.unique(encoded_labels)), random_state=42)
            kmeans_labels = kmeans.fit_predict(X_scaled)

            ari_score = adjusted_rand_score(encoded_labels, kmeans_labels)

            results["mood_cluster_comparison"] = {
                "adjusted_rand_index": float(ari_score),
                "n_ground_truth_clusters": len(np.unique(encoded_labels)),
                "ground_truth_distribution": mood_labels.value_counts().to_dict(),
            }

        return results

    def _plot_feature_distributions(self, df: pd.DataFrame, output_path: Path):
        """Plot feature distributions."""
        plots_dir = output_path / "plots" / "feature_distributions"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Select a subset of features for plotting (to avoid too many plots)
        feature_cols = df.columns[:20]  # First 20 features

        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.flatten()

        for i, col in enumerate(feature_cols):
            if i < len(axes):
                df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                axes[i].set_title(col, fontsize=10)
                axes[i].tick_params(labelsize=8)

        # Hide unused subplots
        for i in range(len(feature_cols), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            plots_dir / "feature_distributions.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_correlation_heatmaps(
        self, pearson_corr: pd.DataFrame, spearman_corr: pd.DataFrame, output_path: Path
    ):
        """Plot correlation heatmaps."""
        plots_dir = output_path / "plots" / "correlations"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Pearson correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
        sns.heatmap(
            pearson_corr,
            mask=mask,
            annot=False,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Pearson Correlation Matrix")
        plt.tight_layout()
        plt.savefig(plots_dir / "pearson_correlation.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Spearman correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            spearman_corr,
            mask=mask,
            annot=False,
            cmap="viridis",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Spearman Correlation Matrix")
        plt.tight_layout()
        plt.savefig(
            plots_dir / "spearman_correlation.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_pca_analysis(
        self,
        explained_variance: np.ndarray,
        cumulative_variance: np.ndarray,
        X_pca: np.ndarray,
        loadings: pd.DataFrame,
        output_path: Path,
    ):
        """Plot PCA analysis results."""
        plots_dir = output_path / "plots" / "pca"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Explained variance plot
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, 21), explained_variance[:20], "bo-")
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.title("Explained Variance by Component")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, 21), cumulative_variance[:20], "ro-")
        plt.axhline(y=0.8, color="k", linestyle="--", label="80% variance")
        plt.axhline(y=0.95, color="g", linestyle="--", label="95% variance")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("Cumulative Explained Variance")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(
            plots_dir / "pca_explained_variance.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # PC1 vs PC2 scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        plt.title("Songs in PC1-PC2 Space")
        plt.grid(True)
        plt.savefig(plots_dir / "pca_scatter.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_clustering_analysis(
        self,
        X_scaled: np.ndarray,
        labels: np.ndarray,
        method_name: str,
        output_path: Path,
    ):
        """Plot clustering results."""
        plots_dir = output_path / "plots" / "clustering"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Use t-SNE for 2D visualization
        try:
            tsne = TSNE(
                n_components=2, random_state=42, perplexity=min(30, len(X_scaled) - 1)
            )
            X_tsne = tsne.fit_transform(X_scaled)

            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap="viridis", alpha=0.6
            )
            plt.colorbar(scatter)
            plt.title(f"{method_name} Clustering Results (t-SNE visualization)")
            plt.xlabel("t-SNE Component 1")
            plt.ylabel("t-SNE Component 2")
            plt.savefig(
                plots_dir / f"{method_name.lower()}_clustering.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        except Exception as e:
            self.logger.warning(f"Could not create t-SNE plot for {method_name}: {e}")

    def _save_analysis_report(self, results: Dict[str, Any], output_path: Path):
        """Save comprehensive analysis report."""
        import json

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Deep convert the results
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(v) for v in obj]
            else:
                return convert_numpy(obj)

        converted_results = deep_convert(results)

        with open(output_path, "w") as f:
            json.dump(converted_results, f, indent=2)

        self.logger.info(f"Analysis report saved to {output_path}")

    def compute_cross_modal_similarity(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute cross-modal similarity as described in SAPPHIRE Sprint 3.
        This addresses the "perceptual gap" by measuring alignment between
        acoustic and lyrical similarity.
        """
        self.logger.info("Computing cross-modal similarity alignment...")

        # Separate acoustic and lyrical features
        acoustic_cols = [
            c
            for c in df.columns
            if any(x in c for x in ["acoustic", "rhythm", "harmony", "quality"])
        ]
        lyrical_cols = [c for c in df.columns if "lyrics" in c]

        if not acoustic_cols or not lyrical_cols:
            self.logger.warning("Insufficient features for cross-modal analysis")
            return {"jaccard_similarity": 0.0, "note": "Insufficient features"}

        # Prepare data
        X_acoustic = df[acoustic_cols].fillna(0)
        X_lyrical = df[lyrical_cols].fillna(0)

        # Standardize
        X_acoustic_scaled = StandardScaler().fit_transform(X_acoustic)
        X_lyrical_scaled = StandardScaler().fit_transform(X_lyrical)

        # Find nearest neighbors for both modalities
        k = min(10, len(df) - 1)  # Number of neighbors

        nn_acoustic = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn_acoustic.fit(X_acoustic_scaled)
        acoustic_neighbors = nn_acoustic.kneighbors(
            X_acoustic_scaled, return_distance=False
        )

        nn_lyrical = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn_lyrical.fit(X_lyrical_scaled)
        lyrical_neighbors = nn_lyrical.kneighbors(
            X_lyrical_scaled, return_distance=False
        )

        # Compute Jaccard similarity between neighbor sets
        jaccard_scores = []
        for i in range(len(df)):
            acoustic_set = set(acoustic_neighbors[i, 1:])  # Exclude self
            lyrical_set = set(lyrical_neighbors[i, 1:])

            intersection = len(acoustic_set.intersection(lyrical_set))
            union = len(acoustic_set.union(lyrical_set))

            jaccard = intersection / union if union > 0 else 0
            jaccard_scores.append(jaccard)

        mean_jaccard = np.mean(jaccard_scores)

        self.logger.info(f"Cross-modal similarity (Jaccard): {mean_jaccard:.4f}")

        return {
            "jaccard_similarity": float(mean_jaccard),
            "jaccard_std": float(np.std(jaccard_scores)),
            "k_neighbors": k,
            "interpretation": self._interpret_jaccard_score(mean_jaccard),
        }

    def _interpret_jaccard_score(self, score: float) -> str:
        """Interpret Jaccard similarity score."""
        if score < 0.1:
            return "Very weak alignment - significant perceptual gap exists"
        elif score < 0.3:
            return "Weak alignment - moderate perceptual gap"
        elif score < 0.5:
            return "Moderate alignment - some perceptual coherence"
        elif score < 0.7:
            return "Good alignment - strong perceptual coherence"
        else:
            return "Excellent alignment - very strong perceptual coherence"
