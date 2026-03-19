"""
Feature cache manager for multi-head MobileNet architectures.

This module provides the FeatureCacheManager class for extracting, caching,
and reusing backbone features with dynamically loaded heads. This enables:
- One-time feature extraction from images
- Dynamic loading/unloading of classification heads
- Memory-efficient inference with multiple heads
- Performance optimization by avoiding redundant backbone computation
"""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

from ..architectures.multi_head_base import MultiHeadMobileNetArchitecture
from ..components.head_configuration import HeadConfiguration


class FeatureCacheManager:
    """Manager for caching backbone features and dynamically loading heads.
    
    This class enables efficient inference by:
    1. Extracting features from the backbone once
    2. Caching features in memory or on disk
    3. Loading/unloading heads dynamically
    4. Running predictions using cached features
    
    Example usage:
        ```python
        # Create cache manager
        cache_manager = FeatureCacheManager(architecture)
        
        # Extract and cache features
        features = cache_manager.extract_features(images)
        
        # Load a head and run prediction
        cache_manager.load_head('person_detection')
        predictions = cache_manager.predict_with_cached_features('person_detection')
        
        # Unload head and load another
        cache_manager.unload_head('person_detection')
        cache_manager.load_head('age_group')
        predictions = cache_manager.predict_with_cached_features('age_group')
        ```
    """
    
    def __init__(self, architecture: MultiHeadMobileNetArchitecture):
        """Initialize the feature cache manager.
        
        Args:
            architecture: Multi-head architecture instance with separable_weights=True
            
        Raises:
            ValueError: If architecture doesn't have separable_weights enabled
        """
        if not architecture.is_separable:
            raise ValueError(
                "FeatureCacheManager requires architecture with separable_weights=True. "
                "Recreate architecture with separable_weights=True in config."
            )
        
        self.architecture = architecture
        self._backbone: Optional[tf.keras.Model] = None
        self._cached_features: Optional[np.ndarray] = None
        self._cached_metadata: Dict[str, Any] = {}
        self._loaded_heads: Dict[str, tf.keras.Model] = {}
    
    @property
    def backbone(self) -> tf.keras.Model:
        """Get the backbone model for feature extraction.
        
        Returns:
            Backbone model
        """
        if self._backbone is None:
            self._backbone = self.architecture.backbone
        return self._backbone
    
    @property
    def feature_shape(self) -> Tuple[int, ...]:
        """Get the shape of features produced by the backbone.
        
        Returns:
            Feature shape (excluding batch dimension)
        """
        dummy_input = tf.zeros((1, *self.architecture.config.input_shape))
        backbone_output = self.backbone(dummy_input, training=False)
        return tuple(backbone_output.shape[1:])
    
    @property
    def cached_features(self) -> Optional[np.ndarray]:
        """Get currently cached features.
        
        Returns:
            Cached features array or None if no features are cached
        """
        return self._cached_features
    
    @property
    def has_cached_features(self) -> bool:
        """Check if features are currently cached.
        
        Returns:
            True if features are cached
        """
        return self._cached_features is not None
    
    @property
    def loaded_head_names(self) -> List[str]:
        """Get list of currently loaded head names.
        
        Returns:
            List of loaded head names
        """
        return list(self._loaded_heads.keys())
    
    def extract_features(self, input_data: np.ndarray, cache: bool = True) -> np.ndarray:
        """Extract features from input data using the backbone.
        
        Args:
            input_data: Input images as numpy array (batch, H, W, C)
            cache: Whether to cache the extracted features
            
        Returns:
            Extracted features (batch, H', W', C')
        """
        # Ensure input is float and normalized if needed
        if input_data.dtype == np.uint8:
            input_data = input_data.astype(np.float32) / 255.0
        
        # Extract features
        features = self.backbone(input_data, training=False)
        features_np = features.numpy()
        
        # Cache if requested
        if cache:
            self._cached_features = features_np
            self._cached_metadata = {
                'input_shape': input_data.shape,
                'feature_shape': features_np.shape,
                'batch_size': input_data.shape[0]
            }
            print(f"Features cached: {features_np.shape}")
        
        return features_np
    
    def save_features(self, filepath: str, input_data: Optional[np.ndarray] = None) -> None:
        """Save features to disk.
        
        If input_data is provided, extract features first. Otherwise, save cached features.
        
        Args:
            filepath: Path to save features (e.g., 'features.npy')
            input_data: Optional input images to extract features from
            
        Raises:
            ValueError: If no features to save
        """
        if input_data is not None:
            features = self.extract_features(input_data, cache=True)
        elif self._cached_features is None:
            raise ValueError("No features to save. Call extract_features() first or provide input_data.")
        else:
            features = self._cached_features
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save features
        np.save(filepath, features)
        
        # Save metadata
        metadata_path = filepath.replace('.npy', '_metadata.pkl')
        metadata = {
            **self._cached_metadata,
            'architecture_config': {
                'input_shape': self.architecture.config.input_shape,
                'alpha': self.architecture.config.arch_params.get('alpha'),
                'head_names': self.architecture.head_names
            }
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Features saved to: {filepath}")
        print(f"Metadata saved to: {metadata_path}")
    
    def load_features(self, filepath: str) -> np.ndarray:
        """Load features from disk.
        
        Args:
            filepath: Path to load features from
            
        Returns:
            Loaded features
            
        Raises:
            FileNotFoundError: If feature file doesn't exist
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Feature file not found: {filepath}")
        
        # Load features
        features = np.load(filepath)
        self._cached_features = features
        
        # Load metadata if available
        metadata_path = filepath.replace('.npy', '_metadata.pkl')
        if Path(metadata_path).exists():
            with open(metadata_path, 'rb') as f:
                self._cached_metadata = pickle.load(f)
            print(f"Features loaded with metadata: {self._cached_metadata}")
        else:
            self._cached_metadata = {
                'feature_shape': features.shape,
                'batch_size': features.shape[0]
            }
            print(f"Features loaded: {features.shape}")
        
        return features
    
    def clear_cache(self) -> None:
        """Clear cached features from memory."""
        self._cached_features = None
        self._cached_metadata = {}
        print("Feature cache cleared")
    
    def build_standalone_head(self, head_name: str) -> tf.keras.Model:
        """Build a standalone head model that takes features as input.
        
        This creates a model that can process cached features directly,
        without needing the backbone.
        
        Args:
            head_name: Name of the head to build
            
        Returns:
            Standalone head model
            
        Raises:
            ValueError: If head not found
        """
        if head_name not in self.architecture.head_names:
            raise ValueError(
                f"Head '{head_name}' not found. "
                f"Available heads: {self.architecture.head_names}"
            )
        
        head_config = self.architecture.get_head_config_by_name(head_name)
        
        # Create input for features
        feature_input = layers.Input(
            shape=self.feature_shape,
            name='feature_input'
        )
        
        # Build head layers
        x = layers.GlobalAveragePooling2D(name=f"{head_name}_gap")(feature_input)
        x = layers.Dropout(head_config.dropout_rate, name=f"{head_name}_drop")(x)
        output = layers.Dense(
            head_config.num_classes,
            activation=head_config.activation,
            name=f"{head_name}_dense"
        )(x)
        
        # Create model
        head_model = Model(
            inputs=feature_input,
            outputs=output,
            name=f"{head_name}_standalone"
        )
        
        return head_model
    
    def load_head(self, head_name: str, weights_path: Optional[str] = None) -> tf.keras.Model:
        """Load a head for use with cached features.
        
        Args:
            head_name: Name of the head to load
            weights_path: Optional path to head weights file
            
        Returns:
            Loaded head model
            
        Raises:
            ValueError: If head not found
        """
        # Build standalone head
        head_model = self.build_standalone_head(head_name)
        
        # Try to copy weights from main model first
        try:
            main_model = self.architecture.get_model()
            for src_name, dst_name in [
                (f"{head_name}_global_pool", f"{head_name}_gap"),
                (f"{head_name}_dropout", f"{head_name}_drop"),
                (f"{head_name}_output", f"{head_name}_dense")
            ]:
                try:
                    src_layer = main_model.get_layer(src_name)
                    dst_layer = head_model.get_layer(dst_name)
                    if src_layer.get_weights():
                        dst_layer.set_weights(src_layer.get_weights())
                except (ValueError, Exception):
                    pass
        except Exception:
            pass  # No weights to copy
        
        # Load from file if provided
        if weights_path and Path(weights_path).exists():
            try:
                head_model.load_weights(weights_path)
                print(f"Head '{head_name}' weights loaded from: {weights_path}")
            except Exception as e:
                print(f"Warning: Could not load weights from {weights_path}: {e}")
        
        # Store in loaded heads
        self._loaded_heads[head_name] = head_model
        print(f"Head '{head_name}' loaded and ready")
        
        return head_model
    
    def unload_head(self, head_name: str) -> None:
        """Unload a head from memory.
        
        Args:
            head_name: Name of the head to unload
        """
        if head_name in self._loaded_heads:
            del self._loaded_heads[head_name]
            print(f"Head '{head_name}' unloaded from memory")
        else:
            print(f"Head '{head_name}' was not loaded")
    
    def unload_all_heads(self) -> None:
        """Unload all heads from memory."""
        head_count = len(self._loaded_heads)
        self._loaded_heads.clear()
        print(f"All {head_count} heads unloaded from memory")
    
    def predict_with_cached_features(
        self,
        head_name: str,
        features: Optional[np.ndarray] = None,
        apply_softmax: bool = True
    ) -> np.ndarray:
        """Run prediction using cached features and a loaded head.
        
        Args:
            head_name: Name of the head to use for prediction
            features: Optional features array (uses cached if None)
            apply_softmax: Whether to apply softmax to outputs
            
        Returns:
            Predictions array
            
        Raises:
            ValueError: If head not loaded or no features available
        """
        # Get features
        if features is None:
            if self._cached_features is None:
                raise ValueError(
                    "No cached features available. "
                    "Call extract_features() first or provide features."
                )
            features = self._cached_features
        
        # Get head model
        if head_name not in self._loaded_heads:
            raise ValueError(
                f"Head '{head_name}' not loaded. "
                f"Call load_head('{head_name}') first."
            )
        
        head_model = self._loaded_heads[head_name]
        
        # Run prediction
        predictions = head_model.predict(features, verbose=0)
        
        # Apply softmax if requested (for logits output)
        if apply_softmax:
            predictions = tf.nn.softmax(predictions).numpy()
        
        return predictions
    
    def predict_all_loaded_heads(
        self,
        features: Optional[np.ndarray] = None,
        apply_softmax: bool = True
    ) -> Dict[str, np.ndarray]:
        """Run predictions using all loaded heads.
        
        Args:
            features: Optional features array (uses cached if None)
            apply_softmax: Whether to apply softmax to outputs
            
        Returns:
            Dictionary mapping head names to predictions
        """
        if not self._loaded_heads:
            raise ValueError("No heads loaded. Call load_head() first.")
        
        predictions = {}
        for head_name in self._loaded_heads:
            predictions[head_name] = self.predict_with_cached_features(
                head_name, features, apply_softmax
            )
        
        return predictions
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information.
        
        Returns:
            Dictionary with memory usage details
        """
        feature_size_mb = 0.0
        if self._cached_features is not None:
            feature_size_mb = self._cached_features.nbytes / (1024 * 1024)
        
        head_info = {}
        total_head_params = 0
        for name, model in self._loaded_heads.items():
            param_count = model.count_params()
            total_head_params += param_count
            head_info[name] = {
                'parameters': param_count,
                'estimated_size_mb': (param_count * 4) / (1024 * 1024)  # FP32
            }
        
        backbone_params = self.backbone.count_params()
        backbone_size_mb = (backbone_params * 4) / (1024 * 1024)
        
        return {
            'cached_features': {
                'size_mb': feature_size_mb,
                'shape': self._cached_features.shape if self._cached_features is not None else None
            },
            'backbone': {
                'parameters': backbone_params,
                'size_mb': backbone_size_mb
            },
            'loaded_heads': head_info,
            'total_heads_loaded': len(self._loaded_heads),
            'total_head_parameters': total_head_params,
            'summary': {
                'features_cached': self.has_cached_features,
                'heads_loaded': len(self._loaded_heads),
                'total_memory_mb': feature_size_mb + sum(
                    h['estimated_size_mb'] for h in head_info.values()
                )
            }
        }
    
    def compare_performance(
        self,
        input_data: np.ndarray,
        head_names: Optional[List[str]] = None,
        num_runs: int = 10
    ) -> Dict[str, Any]:
        """Compare performance: full model vs cached features.
        
        Args:
            input_data: Input images for testing
            head_names: Heads to test (defaults to all)
            num_runs: Number of runs for timing
            
        Returns:
            Performance comparison results
        """
        import time
        
        if head_names is None:
            head_names = self.architecture.head_names
        
        results = {
            'full_model': {},
            'cached_features': {},
            'speedup': {}
        }
        
        # Get full model
        full_model = self.architecture.get_model()
        
        # Warm up
        _ = full_model.predict(input_data[:1], verbose=0)
        
        # Time full model
        start = time.time()
        for _ in range(num_runs):
            _ = full_model.predict(input_data, verbose=0)
        full_model_time = (time.time() - start) / num_runs
        results['full_model']['time_per_batch'] = full_model_time
        
        # Extract features once
        features = self.extract_features(input_data, cache=True)
        
        # Time each head separately
        for head_name in head_names:
            # Load head
            self.load_head(head_name)
            
            # Warm up
            _ = self.predict_with_cached_features(head_name)
            
            # Time head prediction
            start = time.time()
            for _ in range(num_runs):
                _ = self.predict_with_cached_features(head_name)
            head_time = (time.time() - start) / num_runs
            results['cached_features'][head_name] = head_time
            
            # Unload head
            self.unload_head(head_name)
        
        # Calculate speedup
        total_cached_time = sum(results['cached_features'].values())
        results['speedup'] = {
            'full_model_time': full_model_time,
            'total_cached_time': total_cached_time,
            'speedup_factor': full_model_time / total_cached_time if total_cached_time > 0 else 0
        }
        
        return results
    
    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"FeatureCacheManager("
            f"architecture={self.architecture.name}, "
            f"cached={self.has_cached_features}, "
            f"loaded_heads={self.loaded_head_names})"
        )


