# Dataset Statistical Analysis Report

Report generated on: 2025-10-24 22:33:12

This report provides a multi-faceted statistical analysis of the preprocessed song dataset. The goal is to understand the structure and cross-modal relationships within the data to inform the design of a perceptually aware song embedding model.

## Phase 1: Univariate Analysis
This section examines the distribution of individual features to understand their scale, central tendency, and outliers.
### Summary Statistics for Key Features:
```
                                           count         mean           std           min          25%          50%          75%           max
audio_features_duration_seconds            264.0     0.000045  6.789134e-21  4.535147e-05     0.000045     0.000045     0.000045      0.000045
audio_features_snr_percentile_db           264.0     4.883400  6.165097e+00  2.039853e-03     0.971334     2.419265     6.862411     39.125137
audio_features_clipping_percentage         264.0     0.000000  0.000000e+00  0.000000e+00     0.000000     0.000000     0.000000      0.000000
audio_features_rms_energy_mean             264.0     0.000041  2.217990e-04  1.483462e-07     0.000002     0.000006     0.000020      0.003249
audio_features_rms_energy_std              264.0     0.000000  0.000000e+00  0.000000e+00     0.000000     0.000000     0.000000      0.000000
audio_features_zcr_mean                    264.0     0.000072  1.735856e-04  0.000000e+00     0.000000     0.000000     0.000000      0.000488
audio_features_zcr_std                     264.0     0.000000  0.000000e+00  0.000000e+00     0.000000     0.000000     0.000000      0.000000
audio_features_spectral_centroid_mean      264.0  9110.685449  1.579047e+03  8.006391e+03  8071.832884  8311.189472  9528.512135  13941.725691
audio_features_spectral_bandwidth_mean     264.0  5661.134313  3.652317e+02  5.284764e+03  5344.059816  5512.353457  5990.151083   6371.248370
audio_features_mfcc_0_mean                 264.0  -934.115072  1.579144e+02 -1.131371e+03 -1065.753387  -961.022827  -827.653198   -325.338593
audio_features_mfcc_0_std                  264.0     0.000000  0.000000e+00  0.000000e+00     0.000000     0.000000     0.000000      0.000000
audio_features_mfcc_1_mean                 264.0    14.761252  1.435256e+01 -7.595931e+01     9.238311    21.099754    24.915699     26.405533
audio_features_mfcc_1_std                  264.0     0.000000  0.000000e+00  0.000000e+00     0.000000     0.000000     0.000000      0.000000
audio_features_mfcc_2_mean                 264.0   -11.839453  9.403238e+00 -2.108628e+01   -19.651399   -16.066128    -5.577168     15.596002
audio_features_mfcc_2_std                  264.0     0.000000  0.000000e+00  0.000000e+00     0.000000     0.000000     0.000000      0.000000
audio_features_mfcc_3_mean                 264.0     8.838039  6.724157e+00 -7.517076e+00     2.662102    11.463429    14.911789     16.288105
audio_features_mfcc_3_std                  264.0     0.000000  0.000000e+00  0.000000e+00     0.000000     0.000000     0.000000      0.000000
audio_features_mfcc_4_mean                 264.0    -6.722713  5.150700e+00 -1.294189e+01   -11.601279    -8.192034    -1.162127      4.249117
audio_features_mfcc_4_std                  264.0     0.000000  0.000000e+00  0.000000e+00     0.000000     0.000000     0.000000      0.000000
audio_features_mfcc_5_mean                 264.0     5.154783  4.137375e+00 -2.595295e+00     0.343789     5.966297     9.183833     10.494184
audio_features_mfcc_5_std                  264.0     0.000000  0.000000e+00  0.000000e+00     0.000000     0.000000     0.000000      0.000000
audio_features_mfcc_6_mean                 264.0    -4.120305  3.432875e+00 -8.800351e+00    -7.518420    -4.468438    -0.239636      1.578766
audio_features_mfcc_6_std                  264.0     0.000000  0.000000e+00  0.000000e+00     0.000000     0.000000     0.000000      0.000000
audio_features_mfcc_7_mean                 264.0     3.290527  2.907276e+00 -1.025214e+00     0.000000     3.321099     6.192760      7.449301
audio_features_mfcc_7_std                  264.0     0.000000  0.000000e+00  0.000000e+00     0.000000     0.000000     0.000000      0.000000
audio_features_mfcc_8_mean                 264.0    -2.764537  2.513349e+00 -6.509274e+00    -5.275138    -2.638795     0.000000      0.825269
audio_features_mfcc_8_std                  264.0     0.000000  0.000000e+00  0.000000e+00     0.000000     0.000000     0.000000      0.000000
audio_features_mfcc_9_mean                 264.0     2.266478  2.182761e+00 -8.663825e-01     0.000000     1.994013     4.455602      5.650522
audio_features_mfcc_9_std                  264.0     0.000000  0.000000e+00  0.000000e+00     0.000000     0.000000     0.000000      0.000000
audio_features_mfcc_10_mean                264.0    -1.987229  1.932199e+00 -5.082089e+00    -3.924319    -1.669124    -0.023515      0.729501
audio_features_mfcc_10_std                 264.0     0.000000  0.000000e+00  0.000000e+00     0.000000     0.000000     0.000000      0.000000
audio_features_mfcc_11_mean                264.0     1.652503  1.711232e+00 -7.911875e-01     0.000000     1.292933     3.362355      4.484821
audio_features_mfcc_11_std                 264.0     0.000000  0.000000e+00  0.000000e+00     0.000000     0.000000     0.000000      0.000000
audio_features_mfcc_12_mean                264.0    -1.490621  1.545360e+00 -4.114459e+00    -3.025638    -1.123838    -0.032424      0.684393
audio_features_mfcc_12_std                 264.0     0.000000  0.000000e+00  0.000000e+00     0.000000     0.000000     0.000000      0.000000
audio_features_vocal_dominance_score_hpss  264.0     0.500000  8.065687e-08  4.999999e-01     0.500000     0.500000     0.500000      0.500000
lyrics_features_completeness_heuristic     264.0     0.990909  4.173891e-02  8.000000e-01     1.000000     1.000000     1.000000      1.000000
lyrics_features_word_count                 264.0    47.905303  2.949567e+01  6.000000e+00    21.000000    44.500000    67.250000    129.000000
lyrics_features_char_count                 264.0   203.420455  1.251373e+02  2.200000e+01    89.000000   188.500000   285.250000    555.000000
lyrics_features_sentence_count             264.0     1.280303  8.255833e-01  1.000000e+00     1.000000     1.000000     1.000000      7.000000
lyrics_features_type_token_ratio           264.0     0.860700  1.070410e-01  5.384615e-01     0.796086     0.870669     0.943397      1.000000
```

## Phase 2: Correlation Analysis
Examining the relationships between features, both within and across modalities (acoustic vs. lyrical).
Correlation heatmaps have been generated. Key findings:
- High intra-modal correlations might suggest feature redundancy.
- Strong cross-modal correlations are excellent indicators for building a perceptually aware model.

## Phase 3: Principal Component Analysis (PCA)
Using PCA on acoustic features to reduce dimensionality and discover the primary axes of sonic variation in the dataset.
The first two principal components explain **80.83%** of the variance in the acoustic features.
### Top Feature Loadings for PCs:
**PC1 Interpretation:** Features with largest positive loadings: `audio_features_mfcc_5_mean`. Features with largest negative loadings: `audio_features_mfcc_6_mean`.
**PC2 Interpretation:** Features with largest positive loadings: `audio_features_zcr_mean`. Features with largest negative loadings: `audio_features_mfcc_1_mean`.

## Phase 4: K-Means Clustering Analysis
Grouping songs into 5 clusters based on their acoustic properties to see if meaningful lyrical patterns emerge in each group.
Generated violin plots showing the distribution of lyrical features for each acoustic cluster. Check the `plots/3_multivariate` directory to see if, for example, acoustically energetic clusters correspond to lyrically simple songs.

## Phase 5: Cross-Modal Similarity Analysis
This analysis quantifies the alignment between acoustic and lyrical similarity. A high score means sonically similar songs tend to be lyrically similar.
The average Jaccard similarity between the sets of top 10 acoustic and lyrical neighbors is: **0.0320**
A higher score indicates better natural alignment between the sound and the meaning of the songs in the dataset. This is a key baseline metric your embedding model should aim to improve.
