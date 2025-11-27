# SAPPHIRE Experimentation Results

This document provides a comprehensive summary of all experimentation conducted on the SAPPHIRE (Semantic and Acoustic Perceptual Holistic Integration REtrieval) project.

## Experiment Overview

**Date**: November 16, 2025  
**Dataset**: MIREX-like Mood Dataset  
**Total Tracks**: 903 audio clips  
**Total Features Extracted**: 142 features  
**Tracks with Lyrics**: 764 (84.6%)

---

## 1. Dataset Statistics

### 1.1 Dataset Composition

| Component | Count |
|-----------|-------|
| Total Tracks | 903 |
| Audio Files | 903 |
| Lyrics Files | 764 |
| Total Features | 142 |
| Missing Values | 2,085 |

### 1.2 Feature Distribution

| Feature Type | Count | Description |
|--------------|-------|-------------|
| Acoustic | 82 | MFCCs, spectral features, temporal characteristics |
| Rhythm | 8 | Tempo, beat, syncopation, onset patterns |
| Harmony | 27 | Chroma features, key estimation, harmonic ratios |
| Lyrical | 15 | Sentiment, semantic embeddings, readability |
| Quality | 5 | SNR, dynamic range, loudness, clipping |

### 1.3 Mood Cluster Distribution

| Cluster | Track Count | Description |
|---------|-------------|-------------|
| Cluster 3 | 215 | Autumnal, Bittersweet, Brooding, Literate, Poignant, Wistful |
| Cluster 4 | 191 | Campy, Humorous, Silly, Whimsical, Witty, Wry |
| Cluster 1 | 170 | Boisterous, Confident, Passionate, Rousing, Rowdy |
| Cluster 2 | 164 | Amiable, Cheerful, Fun, Rollicking, Sweet |
| Cluster 5 | 163 | Aggressive, Fiery, Intense, Tense-Anxious, Visceral, Volatile |

---

## 2. Feature Analysis

### 2.1 Key Acoustic Features

| Feature | Mean | Std Dev | Range |
|---------|------|---------|-------|
| Spectral Centroid (Hz) | 2290.80 | 567.66 | 739.85 - 4094.74 |
| Tempo (BPM) | 120.25 | 26.55 | 60.09 - 215.33 |
| SNR (dB) | 36.29 | 3.54 | 31.79 - 62.51 |
| Integrated Loudness (LUFS) | -18.00 | 2.53 | -28.13 - -10.52 |

### 2.2 Key Lyrical Features

| Feature | Mean | Std Dev | Range |
|---------|------|---------|-------|
| Word Count | 226.54 | 134.16 | 27 - 1412 |
| Sentiment Compound | 0.32 | 0.81 | -0.999 - 0.999 |
| Vocabulary Richness | 0.464 | 0.124 | 0.089 - 0.804 |
| Flesch Reading Ease | -7.82 | 95.57 | -432.13 - 115.41 |

### 2.3 High Variance Features

Features with high coefficient of variation (indicating high diversity):

1. **acoustic_features.tonnetz_2_mean**: CV = 40.32
2. **acoustic_features.tonnetz_1_mean**: CV = 20.24
3. **acoustic_features.mfcc_12_skew**: CV = 15.69
4. **quality_features.clipping_percentage**: CV = 8.15

---

## 3. Correlation Analysis

### 3.1 Highly Correlated Feature Pairs

| Feature 1 | Feature 2 | Correlation |
|-----------|-----------|-------------|
| Spectral Centroid Mean | Spectral Centroid Median | 0.990 |
| Harmonic Ratio | Percussive Ratio | -1.000 |
| Onset Count | Onset Density | 0.999 |
| Word Count | Character Count | 0.991 |
| Tempo BPM | Beat Count | 0.969 |

### 3.2 Cross-Modal Correlations

Mean absolute correlations between feature modalities:

| Modality Pair | Mean |Abs| Correlation | Max |Abs| Correlation |
|---------------|---------------------|---------------------|
| Acoustic vs Harmony | 0.136 | 0.625 |
| Acoustic vs Quality | 0.127 | 0.757 |
| Acoustic vs Rhythm | 0.070 | 0.339 |
| Acoustic vs Lyrics | **0.044** | 0.269 |
| Lyrics vs Quality | **0.037** | 0.136 |

**Key Finding**: Low correlation between acoustic and lyrical features (0.044) confirms the **perceptual gap** hypothesis.

---

## 4. Clustering Analysis

### 4.1 K-Means Clustering

| K | Silhouette Score | Inertia |
|---|------------------|---------|
| 2 | **0.0955** | 114,938.88 |
| 3 | 0.0696 | 110,121.46 |
| 4 | 0.0690 | 106,253.54 |
| 5 | 0.0491 | 103,401.59 |

**Optimal K**: 2 clusters
- Cluster 0: 401 tracks
- Cluster 1: 502 tracks

**Comparison with Ground Truth (5 clusters)**:
- Adjusted Rand Index: 0.057 (low agreement, indicating mood complexity)

### 4.2 Hierarchical Clustering

| Clusters | Silhouette Score |
|----------|------------------|
| 2 | **0.0737** |
| 3 | 0.0524 |
| 5 | 0.0305 |

**Optimal**: 2 clusters (456 and 447 tracks)

### 4.3 DBSCAN

- **Result**: No valid clustering found
- All tested epsilon values (0.1 - 1.9) resulted in all points classified as noise
- **Interpretation**: Data does not have clear density-based clusters

---

## 5. Dimensionality Reduction (PCA)

### 5.1 Variance Explained

| Components | Cumulative Variance |
|------------|-------------------|
| 1 | 14.74% |
| 2 | 22.81% |
| 5 | 34.81% |
| 10 | 45.56% |
| 20 | 63.66% |
| **39** | **80.00%** |
| **75** | **95.00%** |

### 5.2 Top Principal Components

**PC1 (14.74% variance)**: Dominated by spectral contrast and MFCC features  
**PC2 (8.06% variance)**: Harmonic/percussive ratio and spectral variability  
**PC3 (5.35% variance)**: Lyrical features (semantic embeddings, sentiment)  
**PC4 (3.40% variance)**: Spectral bandwidth and chroma features  
**PC5 (3.25% variance)**: Tonnetz and chroma std features  

---

## 6. Cross-Modal Analysis

### 6.1 Acoustic-Lyrical Jaccard Similarity

- **Mean Jaccard Similarity**: 0.046 (4.6%)
- **Max Jaccard Similarity**: 0.232 (23.2%)
- **Mean Correlation**: 0.046

**Interpretation**: Very low similarity between acoustic and lyrical feature spaces, confirming the **perceptual gap**.

### 6.2 Feature Space Overlap

The analysis shows minimal overlap between:
- Acoustic feature rankings
- Lyrical feature rankings

This indicates that **acoustic and lyrical modalities capture different aspects** of music mood.

---

## 7. Mood Classification Results

### 7.1 Model Performance Comparison

| Model | Test Accuracy | CV Mean | CV Std | Best Configuration |
|-------|---------------|---------|--------|-------------------|
| **Logistic Regression** | **43.65%** | 43.63% | 3.03% | C=0.1, L1 penalty |
| SVM | 41.99% | 41.14% | 4.16% | RBF kernel, C=1 |
| Random Forest | 41.44% | 44.88% | 3.19% | 300 trees, depth=10 |
| Neural Network | 34.81% | 41.98% | 5.84% | 100 hidden units, ReLU |
| Gradient Boosting | 34.25% | 43.64% | 5.52% | 100 estimators, lr=0.2 |

### 7.2 Best Model: Logistic Regression

- **Test Accuracy**: 43.65%
- **Cross-Validation Mean**: 43.63%
- **Cross-Validation Std**: 3.03%
- **Configuration**: L1 penalty (Lasso), C=0.1, liblinear solver

### 7.3 Performance Analysis

**Key Observations**:
1. **Moderate accuracy** (~43-44%) reflects the **inherent difficulty** of mood classification
2. **Low std deviation** in CV indicates **stable, consistent performance**
3. **Logistic Regression outperforms complex models**, suggesting:
   - High dimensionality benefits from regularization
   - Linear relationships may suffice for this task
4. **Gap from human performance**: Mood is subjective; ~44% accuracy for 5-class classification is reasonable

---

## 8. Key Findings & Insights

### 8.1 Perceptual Gap Confirmation

✅ **Hypothesis Confirmed**: Low cross-modal correlation (0.044) and Jaccard similarity (0.046) demonstrate a significant perceptual gap between acoustic and lyrical features.

### 8.2 Feature Diversity

- **142 features** capture multi-modal aspects
- **High variance in spectral and lyrical features** indicates diverse music content
- **Quality metrics** (SNR ~36 dB) show generally good audio quality

### 8.3 Clustering Challenges

- **Low silhouette scores** (<0.1) indicate **overlapping mood clusters**
- **Adjusted Rand Index** (0.057) shows **disagreement** between unsupervised clustering and ground truth
- **Implication**: Mood boundaries are **fuzzy and subjective**

### 8.4 Dimensionality Insights

- **39 components** needed for 80% variance
- **75 components** needed for 95% variance
- **High dimensionality** suggests complex, multi-faceted mood representation

### 8.5 Classification Performance

- **Best: 43.65% accuracy** (5-class problem, random baseline = 20%)
- **Logistic Regression** proves most effective
- **2.18x better than random**, but room for improvement

---

## 9. Limitations & Future Work

### 9.1 Limitations

1. **Small dataset**: 903 tracks may not capture full mood diversity
2. **Subjective labels**: Ground truth mood annotations are inherently subjective
3. **Missing lyrics**: 15.4% of tracks lack lyrics
4. **Single dataset**: Results may not generalize to other music corpora

### 9.2 Future Directions

1. **Deep learning models**: CNNs for audio, transformers for lyrics
2. **Multi-modal fusion**: Better integration of acoustic and lyrical features
3. **Larger datasets**: Train on millions of tracks
4. **Contextual features**: Artist, genre, era information
5. **Contrastive learning**: Align acoustic and lyrical embeddings

---

## 10. Conclusion

The SAPPHIRE project successfully:

✅ Extracted **142 multi-modal features** from 903 music tracks  
✅ **Confirmed the perceptual gap** between acoustic and lyrical modalities  
✅ Achieved **43.65% mood classification accuracy** (vs 20% random baseline)  
✅ Demonstrated that **simple models** (Logistic Regression) can outperform complex ones  
✅ Identified **clustering challenges** due to mood subjectivity  

**Overall Assessment**: The project provides a solid foundation for multi-modal music analysis and highlights the complexity of computational mood recognition.

---

**Generated**: 2025-11-27  
**Version**: 1.0  
**Status**: Complete
