#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from datetime import datetime
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Constants & Configuration ---
REPORT_LINES = [] # Global list to hold lines for the final markdown report

# --- Setup Logging ---
def setup_logging(log_path: Path):
    """Configures logging to both console and a file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )

# --- Core Analysis Functions ---

def load_and_prepare_data(manifest_path: Path) -> pd.DataFrame:
    """Loads the main manifest file and performs initial cleaning."""
    logging.info(f"Loading data from {manifest_path}...")
    try:
        df = pd.read_parquet(manifest_path)
    except Exception as e:
        logging.error(f"Failed to load Parquet file: {e}")
        sys.exit(1)
    
    # Drop columns that are not useful for statistical analysis (e.g., text, paths)
    df = df.drop(columns=[col for col in df.columns if 'cleaned_text' in col or 'path' in col], errors='ignore')
    
    logging.info(f"Data loaded successfully. Shape: {df.shape}")
    return df

def perform_univariate_analysis(df: pd.DataFrame, output_dir: Path, feature_cols: List[str]):
    """Generates plots and summary stats for individual features."""
    logging.info("Starting Univariate Analysis...")
    univariate_path = output_dir / "plots" / "1_univariate"
    univariate_path.mkdir(exist_ok=True)
    
    REPORT_LINES.append("\n## Phase 1: Univariate Analysis\n")
    REPORT_LINES.append("This section examines the distribution of individual features to understand their scale, central tendency, and outliers.\n")

    summary_stats = df[feature_cols].describe().transpose()
    REPORT_LINES.append("### Summary Statistics for Key Features:\n")
    REPORT_LINES.append(f"```\n{summary_stats.to_string()}\n```\n")
    
    for col in tqdm(feature_cols, desc="Analyzing individual features"):
        if df[col].dtype not in ['object', 'bool']:
            plt.figure(figsize=(12, 5))
            
            # Histogram
            plt.subplot(1, 2, 1)
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            
            # Box Plot
            plt.subplot(1, 2, 2)
            sns.boxplot(x=df[col])
            plt.title(f"Box Plot of {col}")
            plt.xlabel(col)
            
            plt.tight_layout()
            plt.savefig(univariate_path / f"{col}_distribution.png")
            plt.close()

    logging.info(f"Univariate plots saved to {univariate_path}")

def perform_correlation_analysis(df: pd.DataFrame, acoustic_cols: List[str], lyrics_cols: List[str], output_dir: Path):
    """Generates correlation heatmaps for different feature domains."""
    logging.info("Starting Correlation Analysis...")
    correlation_path = output_dir / "plots" / "2_correlation"
    correlation_path.mkdir(exist_ok=True)

    REPORT_LINES.append("\n## Phase 2: Correlation Analysis\n")
    REPORT_LINES.append("Examining the relationships between features, both within and across modalities (acoustic vs. lyrical).\n")

    # --- FIX: Select only numeric columns before calculating correlation ---
    numeric_acoustic_df = df[acoustic_cols].select_dtypes(include=np.number)
    numeric_lyrics_df = df[lyrics_cols].select_dtypes(include=np.number)
    # --------------------------------------------------------------------

    # 1. Acoustic Features Correlation
    plt.figure(figsize=(16, 12))
    acoustic_corr = numeric_acoustic_df.corr(method='spearman')
    sns.heatmap(acoustic_corr, annot=False, cmap='coolwarm', fmt=".2f")
    plt.title("Acoustic Feature Correlation (Spearman)")
    plt.tight_layout()
    plt.savefig(correlation_path / "acoustic_features_correlation.png")
    plt.close()

    # 2. Lyrics Features Correlation
    plt.figure(figsize=(10, 8))
    lyrics_corr = numeric_lyrics_df.corr(method='spearman')
    sns.heatmap(lyrics_corr, annot=True, cmap='viridis', fmt=".2f")
    plt.title("Lyrical Feature Correlation (Spearman)")
    plt.tight_layout()
    plt.savefig(correlation_path / "lyrical_features_correlation.png")
    plt.close()
    
    # 3. Cross-Modal Correlation
    cross_corr = pd.concat([numeric_acoustic_df, numeric_lyrics_df], axis=1).corr(method='spearman')
    # Select the rectangle of interest from the full correlation matrix
    cross_corr_subset = cross_corr.loc[numeric_lyrics_df.columns, numeric_acoustic_df.columns]
    plt.figure(figsize=(20, 8))
    sns.heatmap(cross_corr_subset, annot=True, cmap='PRGn', center=0, fmt=".2f", linewidths=.5)
    plt.title("Cross-Modal Correlation: Acoustic vs. Lyrical Features")
    plt.tight_layout()
    plt.savefig(correlation_path / "cross_modal_correlation.png")
    plt.close()
    
    REPORT_LINES.append("Correlation heatmaps have been generated. Key findings:\n")
    REPORT_LINES.append("- High intra-modal correlations might suggest feature redundancy.\n")
    REPORT_LINES.append("- Strong cross-modal correlations are excellent indicators for building a perceptually aware model.\n")
    logging.info(f"Correlation plots saved to {correlation_path}")
    

def perform_pca_analysis(df: pd.DataFrame, acoustic_cols: List[str], output_dir: Path):
    """Performs PCA on acoustic features to find primary axes of variation."""
    logging.info("Starting PCA Analysis on acoustic features...")
    pca_path = output_dir / "plots" / "3_multivariate"
    pca_path.mkdir(exist_ok=True)
    
    REPORT_LINES.append("\n## Phase 3: Principal Component Analysis (PCA)\n")
    REPORT_LINES.append("Using PCA on acoustic features to reduce dimensionality and discover the primary axes of sonic variation in the dataset.\n")

    X = df[acoustic_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_scaled)
    
    # Explained Variance Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
    plt.title("Cumulative Explained Variance by PCA Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True)
    plt.savefig(pca_path / "pca_explained_variance.png")
    plt.close()

    # Scatter Plot of PC1 vs PC2
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], alpha=0.5)
    plt.title("Songs Plotted on First Two Principal Components")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.savefig(pca_path / "pca_pc1_vs_pc2_scatter.png")
    plt.close()

    # Report on PCA loadings
    explained_variance = np.sum(pca.explained_variance_ratio_[:2])
    REPORT_LINES.append(f"The first two principal components explain **{explained_variance:.2%}** of the variance in the acoustic features.\n")
    loadings = pd.DataFrame(pca.components_[:3, :].T, columns=['PC1', 'PC2', 'PC3'], index=acoustic_cols)
    REPORT_LINES.append("### Top Feature Loadings for PCs:\n")
    REPORT_LINES.append(f"**PC1 Interpretation:** Features with largest positive loadings: `{loadings['PC1'].idxmax()}`. Features with largest negative loadings: `{loadings['PC1'].idxmin()}`.\n")
    REPORT_LINES.append(f"**PC2 Interpretation:** Features with largest positive loadings: `{loadings['PC2'].idxmax()}`. Features with largest negative loadings: `{loadings['PC2'].idxmin()}`.\n")

    logging.info(f"PCA plots saved to {pca_path}")

def perform_clustering_analysis(df: pd.DataFrame, acoustic_cols: List[str], lyrics_cols: List[str], output_dir: Path):
    """Clusters songs by acoustic features and analyzes the lyrical properties of each cluster."""
    logging.info("Starting Clustering Analysis...")
    cluster_path = output_dir / "plots" / "3_multivariate"
    cluster_path.mkdir(exist_ok=True)
    
    REPORT_LINES.append("\n## Phase 4: K-Means Clustering Analysis\n")
    REPORT_LINES.append("Grouping songs into 5 clusters based on their acoustic properties to see if meaningful lyrical patterns emerge in each group.\n")

    X = df[acoustic_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['acoustic_cluster'] = kmeans.fit_predict(X_scaled)
    
    for l_col in tqdm(lyrics_cols, desc="Analyzing clusters vs. lyrics"):
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='acoustic_cluster', y=l_col, data=df, palette='viridis', hue='acoustic_cluster', legend=False)
        plt.title(f"Distribution of '{l_col}' Across Acoustic Clusters")
        plt.savefig(cluster_path / f"cluster_vs_{l_col}.png")
        plt.close()
    
    REPORT_LINES.append("Generated violin plots showing the distribution of lyrical features for each acoustic cluster. Check the `plots/3_multivariate` directory to see if, for example, acoustically energetic clusters correspond to lyrically simple songs.\n")
    logging.info(f"Clustering analysis plots saved to {cluster_path}")

def _get_neighbor_overlap(args) -> float:
    """Helper function for parallel processing of neighbor overlap."""
    i, acoustic_neighbors_idx, lyric_neighbors_idx = args
    acoustic_set = set(acoustic_neighbors_idx[i, 1:]) # Exclude self
    lyric_set = set(lyric_neighbors_idx[i, 1:])
    intersection_size = len(acoustic_set.intersection(lyric_set))
    union_size = len(acoustic_set.union(lyric_set))
    return intersection_size / union_size if union_size > 0 else 0

def perform_cross_modal_similarity_analysis(main_df: pd.DataFrame, manifest_path: Path, acoustic_cols: List[str]):
    """Calculates if acoustically similar songs are also lyrically similar."""
    logging.info("Starting Cross-Modal Similarity Analysis (this may take a while)...")
    REPORT_LINES.append("\n## Phase 5: Cross-Modal Similarity Analysis\n")
    REPORT_LINES.append("This analysis quantifies the alignment between acoustic and lyrical similarity. A high score means sonically similar songs tend to be lyrically similar.\n")

    # This is the only place we need the original manifest with text
    df_text = pd.read_parquet(manifest_path, columns=['track_id', 'lyrics_features_cleaned_text'])

    # 1. Prepare Acoustic Features
    X_acoustic = main_df[acoustic_cols].dropna()
    scaler = StandardScaler()
    X_acoustic_scaled = scaler.fit_transform(X_acoustic)

    # 2. Generate Lyrical Embeddings
    logging.info("Generating lyrical embeddings with SentenceTransformer...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    lyrics_list = df_text['lyrics_features_cleaned_text'].tolist()
    # Handle potential None/NaN values
    lyrics_list = [str(lyric) if pd.notna(lyric) else "" for lyric in lyrics_list]

    X_lyrics_embeddings = model.encode(lyrics_list, show_progress_bar=True, batch_size=128)
    
    # 3. Find Nearest Neighbors for both modalities
    K = 10  # Number of neighbors to consider
    logging.info(f"Finding {K} nearest neighbors for both modalities...")
    nn_acoustic = NearestNeighbors(n_neighbors=K+1, metric='euclidean', n_jobs=-1)
    nn_acoustic.fit(X_acoustic_scaled)
    acoustic_neighbors_idx = nn_acoustic.kneighbors(X_acoustic_scaled, return_distance=False)

    nn_lyrics = NearestNeighbors(n_neighbors=K+1, metric='cosine', n_jobs=-1)
    nn_lyrics.fit(X_lyrics_embeddings)
    lyric_neighbors_idx = nn_lyrics.kneighbors(X_lyrics_embeddings, return_distance=False)
    
    # 4. Calculate Overlap in Parallel
    logging.info("Calculating neighbor overlap in parallel...")
    jaccard_scores = []
    
    tasks = [(i, acoustic_neighbors_idx, lyric_neighbors_idx) for i in range(len(main_df))]

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_get_neighbor_overlap, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Calculating Jaccard scores"):
            jaccard_scores.append(future.result())

    mean_jaccard = np.mean(jaccard_scores)
    REPORT_LINES.append(f"The average Jaccard similarity between the sets of top {K} acoustic and lyrical neighbors is: **{mean_jaccard:.4f}**\n")
    REPORT_LINES.append("A higher score indicates better natural alignment between the sound and the meaning of the songs in the dataset. This is a key baseline metric your embedding model should aim to improve.\n")
    logging.info(f"Mean Jaccard Score for neighbor overlap: {mean_jaccard:.4f}")

def generate_summary_report(output_dir: Path):
    """Writes the accumulated report lines to a markdown file."""
    report_path = output_dir / "summary_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Dataset Statistical Analysis Report\n\n")
        f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("This report provides a multi-faceted statistical analysis of the preprocessed song dataset. The goal is to understand the structure and cross-modal relationships within the data to inform the design of a perceptually aware song embedding model.\n")
        f.writelines(REPORT_LINES)
    logging.info(f"Summary report saved to {report_path}")


def main(args):
    """Main orchestrator for the analysis pipeline."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    
    log_path = output_dir / "analysis.log"
    setup_logging(log_path)
    
    start_time = time.time()
    
    # --- Data Loading & Feature Selection ---
    df = load_and_prepare_data(Path(args.manifest_path))
    
    # Define feature groups based on your preprocessing script's output columns
    acoustic_cols = [c for c in df.columns if 'audio_features' in c and 'lufs' not in c]
    lyrics_cols = [c for c in df.columns if 'lyrics_features' in c and 'text' not in c]
    all_feature_cols = acoustic_cols + lyrics_cols
    
    # --- Run Analysis Phases ---
    perform_univariate_analysis(df, output_dir, all_feature_cols)
    perform_correlation_analysis(df, acoustic_cols, lyrics_cols, output_dir)
    perform_pca_analysis(df, acoustic_cols, output_dir)
    perform_clustering_analysis(df, acoustic_cols, lyrics_cols, output_dir)
    
    if not args.skip_similarity:
        perform_cross_modal_similarity_analysis(df, Path(args.manifest_path), acoustic_cols)
    else:
        logging.info("Skipping cross-modal similarity analysis as requested.")
        REPORT_LINES.append("\n## Phase 5: Cross-Modal Similarity Analysis (SKIPPED)\n")

    # --- Final Report ---
    generate_summary_report(output_dir)
    
    duration = time.time() - start_time
    logging.info(f"Analysis complete in {duration:.2f} seconds.")
    logging.info(f"All outputs saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Performs detailed statistical analysis on a preprocessed song dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "manifest_path",
        type=str,
        help="Path to the main 'manifest.parquet' file generated by the preprocessing script.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_report",
        help="Directory to save all plots, logs, and reports.",
    )
    parser.add_argument(
        "--skip-similarity",
        action="store_true",
        help="Skip the computationally intensive cross-modal similarity analysis (which downloads a large model)."
    )
    
    cli_args = parser.parse_args()
    main(cli_args)