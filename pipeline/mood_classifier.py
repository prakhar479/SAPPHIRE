"""
Mood classification module for the SAPPHIRE pipeline.
Implements machine learning models for mood prediction from multi-modal features.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime
import joblib
import warnings

warnings.filterwarnings("ignore")

# Machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, RFE
import matplotlib.pyplot as plt
import seaborn as sns

from .config import config

logger = logging.getLogger(__name__)


class MoodClassifier:
    """
    Multi-modal mood classifier for music analysis.
    Supports multiple ML algorithms and feature selection methods.
    """

    def __init__(self, config_obj=None):
        self.config = config_obj or config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.models = {}
        self.best_model = None
        self.feature_names = None

        # Model configurations
        self.model_configs = {
            "random_forest": {
                "model": RandomForestClassifier,
                "params": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
            },
            "gradient_boosting": {
                "model": GradientBoostingClassifier,
                "params": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                },
            },
            "svm": {
                "model": SVC,
                "params": {
                    "C": [0.1, 1, 10],
                    "kernel": ["rbf", "linear"],
                    "gamma": ["scale", "auto"],
                },
            },
            "logistic_regression": {
                "model": LogisticRegression,
                "params": {
                    "C": [0.1, 1, 10],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear", "saga"],
                },
            },
            "neural_network": {
                "model": MLPClassifier,
                "params": {
                    "hidden_layer_sizes": [(100,), (100, 50), (200, 100)],
                    "activation": ["relu", "tanh"],
                    "alpha": [0.0001, 0.001, 0.01],
                    "learning_rate": ["constant", "adaptive"],
                },
            },
        }

    def prepare_data(
        self, features_df: pd.DataFrame, target_column: str = "mood_cluster"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and targets for training.

        Args:
            features_df: DataFrame with features and target
            target_column: Name of target column

        Returns:
            Tuple of (features, targets)
        """
        self.logger.info("Preparing data for mood classification...")

        # Separate features and targets
        if target_column not in features_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in features")

        # Get feature columns (exclude metadata and target columns)
        exclude_cols = [
            target_column,
            "track_id",
            "mood_category",
            "dataset",
            "language",
        ]
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]

        X = features_df[feature_cols].copy()
        y = features_df[target_column].copy()

        # Defensive: drop identifier-like columns that may leak information
        id_like = [
            c
            for c in X.columns
            if (
                c == "track_id"
                or "file_index" in c
                or c.endswith(".track_id")
                or c.endswith(".file_index")
                or c == "index"
                or c.lower() == "index"
                or c == "extra.track_id"
            )
        ]
        if id_like:
            self.logger.info(
                f"Dropping identifier-like columns to avoid leakage: {id_like}"
            )
            X = X.drop(columns=id_like, errors="ignore")

        # Drop constant columns (no information)
        const_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
        if const_cols:
            self.logger.info(f"Dropping constant columns: {const_cols}")
            X = X.drop(columns=const_cols, errors="ignore")

        # Keep only numeric feature columns (drop any leftover object/meta columns)
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            # Provide helpful debugging info when no numeric features are present
            obj_cols = [c for c in X.columns if X[c].dtype == object]
            raise ValueError(
                f"No numeric feature columns found after filtering. Object-type columns: {obj_cols}"
            )

        # Final safety: remove numeric id-like columns (e.g., file_index may be numeric)
        id_like_numeric = [
            c
            for c in numeric_cols
            if "file_index" in c or c.endswith(".file_index") or c.endswith(".track_id")
        ]
        if id_like_numeric:
            self.logger.info(
                f"Removing numeric identifier-like columns from features: {id_like_numeric}"
            )
            numeric_cols = [c for c in numeric_cols if c not in id_like_numeric]

        X = X[numeric_cols]

        # Handle missing values (numeric only)
        X = X.fillna(X.mean())

        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())

        # Store feature names
        self.feature_names = numeric_cols

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        self.logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        self.logger.info(f"Target classes: {list(self.label_encoder.classes_)}")

        return X.values, y_encoded

    def subset_features_by_importance(
        self, features_df: pd.DataFrame, importance_csv_path: str, top_n: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Subset features using a precomputed importance ranking.

        Returns a tuple of (subset_features_df, used_importance_df).
        """
        importance_path = Path(importance_csv_path)
        if not importance_path.exists():
            self.logger.warning(
                f"Importance CSV not found at {importance_path}, skipping importance-based subsetting"
            )
            return features_df, pd.DataFrame()

        try:
            importance_df = pd.read_csv(importance_path)
        except Exception as e:
            self.logger.warning(
                f"Failed to read importance CSV at {importance_path}: {e}"
            )
            return features_df, pd.DataFrame()

        if "feature" not in importance_df.columns:
            self.logger.warning(
                "Importance CSV does not contain required 'feature' column; skipping importance-based subsetting"
            )
            return features_df, pd.DataFrame()

        if top_n is None or top_n <= 0:
            self.logger.warning(
                f"Invalid top_n value for importance-based subsetting: {top_n}; skipping"
            )
            return features_df, pd.DataFrame()

        self.logger.info(
            f"Applying importance-based feature subset: top {top_n} from {importance_path}"
        )

        # Select top-N features from importance ranking
        selected_df = importance_df.head(top_n).copy()
        selected_features = selected_df["feature"].tolist()

        # Keep only features that actually exist in the DataFrame
        available_mask = selected_df["feature"].isin(features_df.columns)
        used_importance_df = selected_df[available_mask].copy()
        used_features = used_importance_df["feature"].tolist()

        missing_features = selected_df.loc[~available_mask, "feature"].tolist()
        if missing_features:
            self.logger.warning(
                f"The following important features were not found in the feature table and will be ignored: {missing_features}"
            )

        if not used_features:
            self.logger.warning(
                "No overlap between important features and feature table columns; skipping importance-based subsetting"
            )
            return features_df, pd.DataFrame()

        # Always keep key metadata / label columns if present
        base_cols = [
            col
            for col in [
                "track_id",
                "mood_cluster",
                "mood_category",
                "dataset",
                "language",
            ]
            if col in features_df.columns
        ]

        subset_columns = base_cols + used_features
        subset_df = features_df[subset_columns].copy()

        self.logger.info(
            f"Using {len(used_features)} important features (top {top_n}); first few: {used_features[:10]}"
        )

        return subset_df, used_importance_df

    def select_features(
        self, X: np.ndarray, y: np.ndarray, method: str = "mutual_info", k: int = None
    ) -> np.ndarray:
        """
        Perform feature selection.

        Args:
            X: Feature matrix
            y: Target vector
            method: Feature selection method
            k: Number of features to select

        Returns:
            Selected features
        """
        if method == "none":
            # Bypass additional statistical feature selection entirely
            self.logger.info(
                "Feature selection method 'none' specified; using all provided features without additional selection"
            )
            self.feature_selector = None
            return X

        if k is None:
            k = min(self.config.model.max_features or X.shape[1], X.shape[1])

        self.logger.info(f"Selecting {k} features using {method} method...")

        if method == "mutual_info":
            self.feature_selector = SelectKBest(mutual_info_classif, k=k)
        elif method == "f_score":
            self.feature_selector = SelectKBest(f_classif, k=k)
        elif method == "rfe":
            # Use Random Forest for RFE
            estimator = RandomForestClassifier(
                n_estimators=100, random_state=self.config.model.random_state
            )
            self.feature_selector = RFE(estimator, n_features_to_select=k)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")

        X_selected = self.feature_selector.fit_transform(X, y)

        # Get selected feature names
        if hasattr(self.feature_selector, "get_support"):
            selected_mask = self.feature_selector.get_support()
            selected_features = [
                self.feature_names[i]
                for i, selected in enumerate(selected_mask)
                if selected
            ]
            self.logger.info(
                f"Selected features: {selected_features[:10]}..."
            )  # Show first 10

        return X_selected

    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
        """
        Train multiple models with hyperparameter tuning.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Dictionary with model results
        """
        self.logger.info("Training mood classification models...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.model.test_size,
            random_state=self.config.model.random_state,
            stratify=y,
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        results = {}

        for model_name in self.config.model.models:
            if model_name not in self.model_configs:
                self.logger.warning(f"Unknown model: {model_name}")
                continue

            self.logger.info(f"Training {model_name}...")

            try:
                # Get model configuration
                model_config = self.model_configs[model_name]
                model_class = model_config["model"]
                param_grid = model_config["params"]

                # Create base model
                base_model = model_class(random_state=self.config.model.random_state)

                # Perform grid search
                grid_search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=self.config.model.cv_folds,
                    scoring="accuracy",
                    n_jobs=-1,
                    verbose=0,
                )

                grid_search.fit(X_train_scaled, y_train)

                # Get best model
                best_model = grid_search.best_estimator_

                # Evaluate on test set
                y_pred = best_model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)

                # Cross-validation score
                cv_scores = cross_val_score(
                    best_model, X_train_scaled, y_train, cv=self.config.model.cv_folds
                )

                # Store results
                results[model_name] = {
                    "model": best_model,
                    "best_params": grid_search.best_params_,
                    "test_accuracy": accuracy,
                    "cv_mean": cv_scores.mean(),
                    "cv_std": cv_scores.std(),
                    "predictions": y_pred,
                    "true_labels": y_test,
                }

                self.logger.info(
                    f"{model_name} - Test Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
                )

            except Exception as e:
                self.logger.error(f"Error training {model_name}: {e}")
                continue

        # Store models and find best one
        self.models = results
        if results:
            best_model_name = max(
                results.keys(), key=lambda k: results[k]["test_accuracy"]
            )
            self.best_model = results[best_model_name]["model"]
            self.logger.info(
                f"Best model: {best_model_name} (Accuracy: {results[best_model_name]['test_accuracy']:.4f})"
            )

        return results

    def evaluate_models(self, results: Dict[str, Dict], output_dir: str):
        """
        Generate comprehensive evaluation reports.

        Args:
            results: Model training results
            output_dir: Directory to save evaluation results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info("Generating model evaluation reports...")

        # Create comparison DataFrame
        comparison_data = []
        for model_name, result in results.items():
            comparison_data.append(
                {
                    "Model": model_name,
                    "Test Accuracy": result["test_accuracy"],
                    "CV Mean": result["cv_mean"],
                    "CV Std": result["cv_std"],
                    "Best Params": str(result["best_params"]),
                }
            )

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(output_path / "model_comparison.csv", index=False)

        # Generate detailed reports for each model
        for model_name, result in results.items():
            model_dir = output_path / model_name
            model_dir.mkdir(exist_ok=True)

            # Classification report
            y_true = result["true_labels"]
            y_pred = result["predictions"]

            # Convert back to original labels
            true_labels = self.label_encoder.inverse_transform(y_true)
            pred_labels = self.label_encoder.inverse_transform(y_pred)

            # Generate classification report
            report = classification_report(true_labels, pred_labels, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(model_dir / "classification_report.csv")

            # Confusion matrix
            cm = confusion_matrix(true_labels, pred_labels)

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=self.label_encoder.classes_,
                yticklabels=self.label_encoder.classes_,
            )
            plt.title(f"Confusion Matrix - {model_name}")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.tight_layout()
            plt.savefig(
                model_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        # Model comparison plot
        plt.figure(figsize=(12, 6))
        models = list(results.keys())
        accuracies = [results[m]["test_accuracy"] for m in models]
        cv_means = [results[m]["cv_mean"] for m in models]
        cv_stds = [results[m]["cv_std"] for m in models]

        x = np.arange(len(models))
        width = 0.35

        plt.bar(x - width / 2, accuracies, width, label="Test Accuracy", alpha=0.8)
        plt.errorbar(
            x + width / 2,
            cv_means,
            yerr=cv_stds,
            fmt="o",
            label="CV Mean ± Std",
            capsize=5,
        )

        plt.xlabel("Models")
        plt.ylabel("Accuracy")
        plt.title("Model Performance Comparison")
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / "model_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Evaluation reports saved to {output_path}")

    def get_feature_importance(self, model_name: str = None) -> pd.DataFrame:
        """
        Get feature importance from the best model.

        Args:
            model_name: Specific model name, or None for best model

        Returns:
            DataFrame with feature importance
        """
        if model_name is None:
            model = self.best_model
            model_name = "best_model"
        else:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            model = self.models[model_name]["model"]

        if not hasattr(model, "feature_importances_"):
            self.logger.warning(
                f"Model {model_name} does not have feature_importances_ attribute"
            )
            return pd.DataFrame()

        # Get selected feature names
        if self.feature_selector and hasattr(self.feature_selector, "get_support"):
            selected_mask = self.feature_selector.get_support()
            selected_features = [
                self.feature_names[i]
                for i, selected in enumerate(selected_mask)
                if selected
            ]
        else:
            selected_features = self.feature_names

        importance_df = pd.DataFrame(
            {"feature": selected_features, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        return importance_df

    def predict(
        self, features: Union[np.ndarray, pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the best model.

        Args:
            features: Feature matrix or DataFrame

        Returns:
            Tuple of (predicted_labels, prediction_probabilities)
        """
        if self.best_model is None:
            raise ValueError("No trained model available. Train models first.")

        # Convert DataFrame to array if needed
        if isinstance(features, pd.DataFrame):
            if self.feature_names:
                # Select only the features used during training
                available_features = [
                    f for f in self.feature_names if f in features.columns
                ]
                features = features[available_features].values
            else:
                features = features.values

        # Apply feature selection if used
        if self.feature_selector:
            features = self.feature_selector.transform(features)

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Make predictions
        predictions = self.best_model.predict(features_scaled)

        # Get probabilities if available
        if hasattr(self.best_model, "predict_proba"):
            probabilities = self.best_model.predict_proba(features_scaled)
        else:
            probabilities = None

        # Convert back to original labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)

        return predicted_labels, probabilities

    def save_model(self, filepath: str):
        """Save the trained model and preprocessing components."""
        model_data = {
            "best_model": self.best_model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_selector": self.feature_selector,
            "feature_names": self.feature_names,
            "models": self.models,
        }

        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model and preprocessing components."""
        model_data = joblib.load(filepath)

        self.best_model = model_data["best_model"]
        self.scaler = model_data["scaler"]
        self.label_encoder = model_data["label_encoder"]
        self.feature_selector = model_data["feature_selector"]
        self.feature_names = model_data["feature_names"]
        self.models = model_data["models"]

        self.logger.info(f"Model loaded from {filepath}")

    def cross_modal_analysis(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze cross-modal relationships between acoustic and lyrical features.

        Args:
            features_df: DataFrame with multi-modal features

        Returns:
            Dictionary with cross-modal analysis results
        """
        self.logger.info("Performing cross-modal analysis...")

        # Identify acoustic and lyrical features
        acoustic_features = [
            col
            for col in features_df.columns
            if any(
                keyword in col.lower()
                for keyword in ["mfcc", "spectral", "chroma", "tempo", "rhythm"]
            )
        ]

        lyrical_features = [
            col
            for col in features_df.columns
            if any(
                keyword in col.lower()
                for keyword in ["sentiment", "embedding", "readability", "lyrics"]
            )
        ]

        results = {
            "acoustic_features": acoustic_features,
            "lyrical_features": lyrical_features,
            "correlations": {},
            "jaccard_similarity": {},
        }

        if acoustic_features and lyrical_features:
            # Calculate correlations between acoustic and lyrical features
            acoustic_data = features_df[acoustic_features].fillna(0)
            lyrical_data = features_df[lyrical_features].fillna(0)

            # Cross-correlation matrix
            cross_corr = np.corrcoef(acoustic_data.T, lyrical_data.T)
            n_acoustic = len(acoustic_features)
            cross_corr_subset = cross_corr[:n_acoustic, n_acoustic:]

            results["correlations"] = {
                "matrix": cross_corr_subset,
                "acoustic_features": acoustic_features,
                "lyrical_features": lyrical_features,
                "max_correlation": np.max(np.abs(cross_corr_subset)),
                "mean_correlation": np.mean(np.abs(cross_corr_subset)),
            }

            # Jaccard similarity analysis (as mentioned in Sprint 3)
            # Binarize features for Jaccard similarity
            acoustic_binary = (acoustic_data > acoustic_data.median()).astype(int)
            lyrical_binary = (lyrical_data > lyrical_data.median()).astype(int)

            jaccard_scores = []
            for i in range(len(features_df)):
                # Calculate Jaccard similarity for each track
                a_features = set(np.where(acoustic_binary.iloc[i] == 1)[0])
                l_features = set(np.where(lyrical_binary.iloc[i] == 1)[0])

                if len(a_features) == 0 and len(l_features) == 0:
                    jaccard = 1.0
                else:
                    intersection = len(a_features.intersection(l_features))
                    union = len(a_features.union(l_features))
                    jaccard = intersection / union if union > 0 else 0.0

                jaccard_scores.append(jaccard)

            results["jaccard_similarity"] = {
                "scores": jaccard_scores,
                "mean": np.mean(jaccard_scores),
                "std": np.std(jaccard_scores),
                "min": np.min(jaccard_scores),
                "max": np.max(jaccard_scores),
            }

            self.logger.info(
                f"Cross-modal analysis complete. Mean Jaccard similarity: {np.mean(jaccard_scores):.4f}"
            )

        return results
