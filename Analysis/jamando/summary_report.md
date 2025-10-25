# Dataset Statistical Analysis Report

Report generated on: 2025-10-24 22:46:52

This report provides a multi-faceted statistical analysis of the preprocessed song dataset. The goal is to understand the structure and cross-modal relationships within the data to inform the design of a perceptually aware song embedding model.

## Phase 1: Univariate Analysis
This section examines the distribution of individual features to understand their scale, central tendency, and outliers.
### Summary Statistics for Key Features:
```
                                           count          mean         std          min          25%          50%          75%          max
audio_features_duration_seconds             79.0  2.151604e+02   37.820204   147.795011   189.120000   212.036689   235.130454   336.000000
audio_features_snr_percentile_db            79.0  1.016583e+01    1.104462     8.015605     9.529069     9.960247    10.789337    14.599831
audio_features_clipping_percentage          79.0  4.570615e-07    0.000004     0.000000     0.000000     0.000000     0.000000     0.000036
audio_features_rms_energy_mean              79.0  6.946191e-02    0.005920     0.044608     0.066381     0.069751     0.073993     0.083578
audio_features_rms_energy_std               79.0  3.068030e-02    0.007424     0.017635     0.025222     0.029487     0.035469     0.054887
audio_features_zcr_mean                     79.0  5.148597e-02    0.017685     0.016154     0.041691     0.049786     0.060908     0.113330
audio_features_zcr_std                      79.0  3.975325e-02    0.012672     0.016706     0.030418     0.036893     0.046381     0.071943
audio_features_spectral_centroid_mean       79.0  2.106217e+03  457.477266   914.931042  1827.954555  2132.226218  2397.383836  3509.274755
audio_features_spectral_bandwidth_mean      79.0  2.319061e+03  279.720109  1327.794464  2162.117461  2349.201063  2480.403947  2867.999785
audio_features_mfcc_0_mean                  79.0 -2.640249e+02   29.620095  -350.036200  -282.703488  -262.442058  -241.973704  -205.157075
audio_features_mfcc_0_std                   79.0  7.753412e+01   15.567212    35.460202    67.289168    78.468889    88.148493   118.029340
audio_features_mfcc_1_mean                  79.0  1.634420e+02   14.044058   118.882576   155.450889   165.844056   173.320630   190.483188
audio_features_mfcc_1_std                   79.0  3.117015e+01    6.128504    16.772033    27.293147    30.937305    36.088246    45.452589
audio_features_mfcc_2_mean                  79.0 -3.802816e+01   23.905226   -90.545134   -52.645913   -37.969839   -22.235121    31.465372
audio_features_mfcc_2_std                   79.0  2.971050e+01    6.974665    11.276356    24.290247    29.580940    34.492624    45.506828
audio_features_mfcc_3_mean                  79.0  5.491495e+01   13.240938     7.471196    48.878270    55.887713    62.463983    85.362103
audio_features_mfcc_3_std                   79.0  2.449557e+01    4.806338    12.634068    21.452347    23.712809    27.429688    36.417612
audio_features_mfcc_4_mean                  79.0 -4.392727e+00    9.214197   -27.877620    -8.704811    -4.582839     0.303308    27.529467
audio_features_mfcc_4_std                   79.0  1.895465e+01    3.050427    11.562417    17.397444    18.817608    20.942863    26.342664
audio_features_mfcc_5_mean                  79.0  1.507667e+01    5.588259     2.619605    11.574176    15.149158    18.015097    31.626868
audio_features_mfcc_5_std                   79.0  1.294939e+01    2.533290     7.969144    10.845895    12.721831    14.968091    19.803980
audio_features_mfcc_6_mean                  79.0  8.908751e+00    5.143418    -7.724607     6.090995     9.151053    12.224878    25.479398
audio_features_mfcc_6_std                   79.0  1.093311e+01    2.247600     6.784670     9.393117    10.604224    12.402981    17.118693
audio_features_mfcc_7_mean                  79.0 -7.388095e+00    4.608910   -20.947869    -9.899902    -7.054335    -5.254008     7.142015
audio_features_mfcc_7_std                   79.0  1.100403e+01    1.781937     7.561497    10.043163    10.789398    11.956288    15.894341
audio_features_mfcc_8_mean                  79.0  1.628606e+01    5.571104    -4.453837    14.062122    17.085133    19.800852    27.978563
audio_features_mfcc_8_std                   79.0  1.139742e+01    2.134793     7.512730     9.815413    11.104315    12.836649    16.837959
audio_features_mfcc_9_mean                  79.0 -1.140445e+01    4.283135   -22.859932   -14.664414   -11.344919    -8.948684    -2.039788
audio_features_mfcc_9_std                   79.0  1.032770e+01    1.688419     7.099133     9.183115    10.091579    11.448771    14.940594
audio_features_mfcc_10_mean                 79.0  7.647459e+00    5.178290    -9.049494     4.682674     8.127897    10.181166    20.955478
audio_features_mfcc_10_std                  79.0  9.734860e+00    1.851388     6.681021     8.436741     9.532686    10.750166    14.556244
audio_features_mfcc_11_mean                 79.0 -2.062641e-01    3.110894    -8.366953    -2.317252     0.175124     1.470964     7.762896
audio_features_mfcc_11_std                  79.0  8.433473e+00    1.355083     6.131465     7.377465     8.368985     9.337742    12.084143
audio_features_mfcc_12_mean                 79.0 -3.268916e+00    3.191373   -12.073677    -5.057964    -3.209923    -1.536487     3.344213
audio_features_mfcc_12_std                  79.0  8.083831e+00    1.234501     5.750058     7.078395     8.347363     8.875058    11.149478
audio_features_vocal_dominance_score_hpss   79.0  8.107933e-01    0.076355     0.560229     0.768076     0.811137     0.868715     0.945864
lyrics_features_completeness_heuristic      79.0  1.000000e+00    0.000000     1.000000     1.000000     1.000000     1.000000     1.000000
lyrics_features_word_count                  79.0  2.741266e+02  127.586006    82.000000   193.000000   252.000000   335.000000   643.000000
lyrics_features_char_count                  79.0  1.484152e+03  665.005361   470.000000  1016.000000  1313.000000  1863.000000  3314.000000
lyrics_features_sentence_count              79.0  2.810127e+00    3.092903     1.000000     1.000000     1.000000     3.000000    17.000000
lyrics_features_type_token_ratio            79.0  4.830868e-01    0.129083     0.195238     0.415126     0.456897     0.558294     0.902439
```

## Phase 2: Correlation Analysis
Examining the relationships between features, both within and across modalities (acoustic vs. lyrical).
Correlation heatmaps have been generated. Key findings:
- High intra-modal correlations might suggest feature redundancy.
- Strong cross-modal correlations are excellent indicators for building a perceptually aware model.

## Phase 3: Principal Component Analysis (PCA)
Using PCA on acoustic features to reduce dimensionality and discover the primary axes of sonic variation in the dataset.
The first two principal components explain **46.04%** of the variance in the acoustic features.
### Top Feature Loadings for PCs:
**PC1 Interpretation:** Features with largest positive loadings: `audio_features_mfcc_12_std`. Features with largest negative loadings: `audio_features_mfcc_0_mean`.
**PC2 Interpretation:** Features with largest positive loadings: `audio_features_spectral_centroid_mean`. Features with largest negative loadings: `audio_features_mfcc_4_mean`.

## Phase 4: K-Means Clustering Analysis
Grouping songs into 5 clusters based on their acoustic properties to see if meaningful lyrical patterns emerge in each group.
Generated violin plots showing the distribution of lyrical features for each acoustic cluster. Check the `plots/3_multivariate` directory to see if, for example, acoustically energetic clusters correspond to lyrically simple songs.

## Phase 5: Cross-Modal Similarity Analysis
This analysis quantifies the alignment between acoustic and lyrical similarity. A high score means sonically similar songs tend to be lyrically similar.
The average Jaccard similarity between the sets of top 10 acoustic and lyrical neighbors is: **0.0755**
A higher score indicates better natural alignment between the sound and the meaning of the songs in the dataset. This is a key baseline metric your embedding model should aim to improve.
