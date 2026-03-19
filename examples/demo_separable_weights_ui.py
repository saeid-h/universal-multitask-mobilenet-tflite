#!/usr/bin/env python3
"""
Gradio Web UI for Separable Weights Demo.

This demo showcases the separable weights feature for multi-head MobileNet V3 models
with a web-based interface. It demonstrates:
- Feature extraction and caching
- Dynamic head loading/unloading
- Multi-head inference
- Memory usage visualization
- Performance comparison

Usage:
    python demo_separable_weights_ui.py

Then open the displayed URL in your browser.
"""

import sys
from pathlib import Path
import numpy as np
import tempfile
import time
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import gradio as gr
import tensorflow as tf

from models.components.multi_head_model_config import MultiHeadModelConfig
from models.components.head_configuration import create_head_config_from_list
from models.architectures.mobilenet_v3_qat_multi import MultiHeadMobileNetV3QATArchitecture
from models.utils import FeatureCacheManager


# Global state
class AppState:
    """Global application state."""
    architecture: Optional[MultiHeadMobileNetV3QATArchitecture] = None
    cache_manager: Optional[FeatureCacheManager] = None
    temp_dir: str = tempfile.mkdtemp(prefix="separable_weights_ui_")


state = AppState()


def create_model(
    input_height: int,
    input_width: int,
    channels: int,
    alpha: float,
    head_configs_str: str
) -> str:
    """Create a model with the given configuration."""
    try:
        # Parse head configurations
        # Format: "head_name:num_classes, head_name2:num_classes2"
        head_parts = [h.strip() for h in head_configs_str.split(',')]
        head_names = []
        head_classes = []
        
        for part in head_parts:
            if ':' in part:
                name, num = part.split(':')
                head_names.append(name.strip())
                head_classes.append(int(num.strip()))
            else:
                head_classes.append(int(part.strip()))
        
        if not head_names:
            head_names = None
        
        # Create configuration
        head_configs = create_head_config_from_list(head_classes, head_names)
        
        config = MultiHeadModelConfig(
            input_shape=(input_height, input_width, channels),
            head_configs=head_configs,
            arch_params={
                'alpha': alpha,
                'use_pretrained': False,
            },
            training_mode='joint',
            separable_weights=True
        )
        
        # Create architecture
        state.architecture = MultiHeadMobileNetV3QATArchitecture(config)
        model = state.architecture.get_model()
        
        # Create cache manager
        state.cache_manager = FeatureCacheManager(state.architecture)
        
        # Build info string
        info = [
            f"✅ Model created successfully!",
            f"",
            f"**Configuration:**",
            f"- Input shape: {state.architecture.config.input_shape}",
            f"- Alpha: {alpha}",
            f"- Total parameters: {model.count_params():,}",
            f"- Separable weights: Enabled",
            f"",
            f"**Heads:**"
        ]
        
        for head in state.architecture.head_configs:
            info.append(f"- {head.name}: {head.num_classes} classes")
        
        return "\n".join(info)
        
    except Exception as e:
        return f"❌ Error creating model: {str(e)}"


def get_model_info() -> str:
    """Get current model information."""
    if state.architecture is None:
        return "No model loaded. Create a model first."
    
    model = state.architecture.get_model()
    
    info = [
        f"**Model: {state.architecture.name}**",
        f"",
        f"**Configuration:**",
        f"- Input shape: {state.architecture.config.input_shape}",
        f"- Total parameters: {model.count_params():,}",
        f"- Feature shape: {state.cache_manager.feature_shape}",
        f"",
        f"**Heads:**"
    ]
    
    for head in state.architecture.head_configs:
        info.append(f"- {head.name}: {head.num_classes} classes")
    
    # Cache status
    info.append("")
    info.append("**Cache Status:**")
    info.append(f"- Features cached: {state.cache_manager.has_cached_features}")
    if state.cache_manager.has_cached_features:
        info.append(f"- Cached shape: {state.cache_manager.cached_features.shape}")
    info.append(f"- Loaded heads: {state.cache_manager.loaded_head_names or 'None'}")
    
    return "\n".join(info)


def process_image(image: np.ndarray) -> Tuple[str, np.ndarray]:
    """Process an uploaded image and extract features."""
    if state.architecture is None:
        return "❌ No model loaded. Create a model first.", None
    
    if image is None:
        return "❌ No image provided.", None
    
    try:
        # Get expected input shape
        h, w, c = state.architecture.config.input_shape
        
        # Resize image
        image_resized = tf.image.resize(image, (h, w)).numpy()
        
        # Normalize to [0, 1]
        if image_resized.max() > 1.0:
            image_resized = image_resized / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_resized, axis=0).astype(np.float32)
        
        # Extract features
        features = state.cache_manager.extract_features(image_batch)
        
        info = [
            f"✅ Features extracted!",
            f"",
            f"**Image Processing:**",
            f"- Original shape: {image.shape}",
            f"- Resized to: {image_batch.shape[1:]}",
            f"",
            f"**Features:**",
            f"- Shape: {features.shape}",
            f"- Min: {features.min():.4f}",
            f"- Max: {features.max():.4f}",
            f"- Mean: {features.mean():.4f}"
        ]
        
        return "\n".join(info), image_resized
        
    except Exception as e:
        return f"❌ Error processing image: {str(e)}", None


def run_inference(selected_heads: List[str]) -> str:
    """Run inference on cached features with selected heads."""
    if state.architecture is None:
        return "❌ No model loaded. Create a model first."
    
    if not state.cache_manager.has_cached_features:
        return "❌ No features cached. Upload an image first."
    
    if not selected_heads:
        return "❌ No heads selected. Select at least one head."
    
    try:
        results = []
        results.append("**Inference Results:**\n")
        
        for head_name in selected_heads:
            # Load head
            state.cache_manager.load_head(head_name)
            
            # Run prediction
            predictions = state.cache_manager.predict_with_cached_features(head_name)
            
            # Get class prediction
            class_idx = np.argmax(predictions[0])
            confidence = predictions[0][class_idx]
            
            results.append(f"**{head_name}:**")
            results.append(f"- Predicted class: {class_idx}")
            results.append(f"- Confidence: {confidence:.4f}")
            results.append(f"- All probabilities: {predictions[0].round(4).tolist()}")
            results.append("")
            
            # Unload head
            state.cache_manager.unload_head(head_name)
        
        return "\n".join(results)
        
    except Exception as e:
        return f"❌ Error running inference: {str(e)}"


def get_head_choices() -> List[str]:
    """Get list of available heads."""
    if state.architecture is None:
        return []
    return state.architecture.head_names


def save_weights() -> str:
    """Save backbone and head weights separately."""
    if state.architecture is None:
        return "❌ No model loaded. Create a model first."
    
    try:
        # Save to temp directory
        weights_path = Path(state.temp_dir) / "weights"
        saved_files = state.architecture.save_all_weights_separately(str(weights_path))
        
        info = [
            f"✅ Weights saved!",
            f"",
            f"**Saved files:**"
        ]
        
        for name, path in saved_files.items():
            size_kb = Path(path).stat().st_size / 1024
            info.append(f"- {name}: {size_kb:.1f} KB")
        
        info.append(f"")
        info.append(f"**Location:** {weights_path}")
        
        return "\n".join(info)
        
    except Exception as e:
        return f"❌ Error saving weights: {str(e)}"


def get_memory_info() -> str:
    """Get memory usage information."""
    if state.cache_manager is None:
        return "❌ No model loaded. Create a model first."
    
    try:
        # Load all heads to get full memory picture
        for head_name in state.architecture.head_names:
            if head_name not in state.cache_manager.loaded_head_names:
                state.cache_manager.load_head(head_name)
        
        memory_info = state.cache_manager.get_memory_usage()
        
        info = [
            f"**Memory Usage:**",
            f"",
            f"**Backbone:**",
            f"- Parameters: {memory_info['backbone']['parameters']:,}",
            f"- Size: {memory_info['backbone']['size_mb']:.2f} MB",
            f"",
            f"**Cached Features:**",
            f"- Size: {memory_info['cached_features']['size_mb']:.2f} MB",
        ]
        
        if memory_info['cached_features']['shape']:
            info.append(f"- Shape: {memory_info['cached_features']['shape']}")
        
        info.append("")
        info.append("**Loaded Heads:**")
        
        for head_name, head_info in memory_info['loaded_heads'].items():
            info.append(f"- {head_name}: {head_info['parameters']:,} params ({head_info['estimated_size_mb']:.3f} MB)")
        
        info.append("")
        info.append(f"**Total:** {memory_info['summary']['total_memory_mb']:.2f} MB")
        
        return "\n".join(info)
        
    except Exception as e:
        return f"❌ Error getting memory info: {str(e)}"


def compare_performance() -> str:
    """Compare performance between full model and cached features."""
    if state.architecture is None:
        return "❌ No model loaded. Create a model first."
    
    try:
        # Generate test batch
        h, w, c = state.architecture.config.input_shape
        test_batch = np.random.rand(4, h, w, c).astype(np.float32)
        
        # Full model timing
        full_model = state.architecture.get_model()
        
        # Warm up
        _ = full_model.predict(test_batch[:1], verbose=0)
        
        # Time full model
        start = time.time()
        for _ in range(10):
            _ = full_model.predict(test_batch, verbose=0)
        full_time = (time.time() - start) / 10 * 1000  # ms
        
        # Cached features timing
        features = state.cache_manager.extract_features(test_batch, cache=True)
        
        head_times = {}
        for head_name in state.architecture.head_names[:2]:  # Test first 2 heads
            state.cache_manager.load_head(head_name)
            
            # Warm up
            _ = state.cache_manager.predict_with_cached_features(head_name)
            
            # Time
            start = time.time()
            for _ in range(10):
                _ = state.cache_manager.predict_with_cached_features(head_name)
            head_times[head_name] = (time.time() - start) / 10 * 1000  # ms
            
            state.cache_manager.unload_head(head_name)
        
        # Calculate savings
        total_cached_time = sum(head_times.values())
        speedup = full_time / total_cached_time if total_cached_time > 0 else 0
        
        info = [
            f"**Performance Comparison:**",
            f"",
            f"**Full Model (all heads at once):**",
            f"- Time per batch: {full_time:.2f} ms",
            f"",
            f"**Cached Features (per head):**"
        ]
        
        for head_name, head_time in head_times.items():
            info.append(f"- {head_name}: {head_time:.2f} ms")
        
        info.append("")
        info.append(f"**Total cached time:** {total_cached_time:.2f} ms")
        info.append(f"**Speedup:** {speedup:.2f}x")
        info.append("")
        info.append("*Note: Speedup increases when running fewer heads than the full model.*")
        
        return "\n".join(info)
        
    except Exception as e:
        return f"❌ Error comparing performance: {str(e)}"


def create_ui():
    """Create the Gradio interface."""
    
    with gr.Blocks(
        title="Separable Weights Demo",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("""
        # 🧠 Separable Weights Demo
        
        This demo showcases the separable weights feature for multi-head MobileNet V3 models.
        
        **Features:**
        - Create models with separable backbone and heads
        - Extract and cache features
        - Run inference with dynamically loaded heads
        - Compare performance and memory usage
        """)
        
        with gr.Tab("📦 Model Setup"):
            gr.Markdown("## Create Model")
            
            with gr.Row():
                with gr.Column():
                    input_height = gr.Slider(
                        minimum=32, maximum=512, value=128, step=32,
                        label="Input Height"
                    )
                    input_width = gr.Slider(
                        minimum=32, maximum=512, value=128, step=32,
                        label="Input Width"
                    )
                    channels = gr.Radio(
                        choices=[1, 3], value=3,
                        label="Channels (1=Grayscale, 3=RGB)"
                    )
                    alpha = gr.Dropdown(
                        choices=[0.25, 0.50, 0.75, 1.0], value=0.25,
                        label="Alpha (Width Multiplier)"
                    )
                    head_configs = gr.Textbox(
                        value="object:5, person:2, age:3",
                        label="Head Configurations",
                        info="Format: name:classes, name:classes (or just: 5, 2, 3)"
                    )
                
                with gr.Column():
                    create_btn = gr.Button("Create Model", variant="primary")
                    model_info = gr.Markdown("No model created yet.")
            
            create_btn.click(
                fn=create_model,
                inputs=[input_height, input_width, channels, alpha, head_configs],
                outputs=model_info
            )
            
            gr.Markdown("---")
            gr.Markdown("## Save Weights")
            
            with gr.Row():
                save_btn = gr.Button("Save Backbone & Head Weights", variant="secondary")
                save_info = gr.Markdown("")
            
            save_btn.click(fn=save_weights, outputs=save_info)
        
        with gr.Tab("🖼️ Feature Extraction"):
            gr.Markdown("## Upload Image & Extract Features")
            
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(
                        label="Upload Image",
                        type="numpy"
                    )
                    extract_btn = gr.Button("Extract Features", variant="primary")
                
                with gr.Column():
                    processed_image = gr.Image(label="Processed Image")
                    extract_info = gr.Markdown("")
            
            extract_btn.click(
                fn=process_image,
                inputs=image_input,
                outputs=[extract_info, processed_image]
            )
        
        with gr.Tab("🎯 Inference"):
            gr.Markdown("## Run Inference on Cached Features")
            
            with gr.Row():
                with gr.Column():
                    head_selector = gr.CheckboxGroup(
                        choices=[],
                        label="Select Heads for Inference",
                        info="Choose which heads to run"
                    )
                    refresh_heads_btn = gr.Button("Refresh Head List")
                    run_btn = gr.Button("Run Inference", variant="primary")
                
                with gr.Column():
                    inference_results = gr.Markdown("")
            
            def refresh_heads():
                return gr.update(choices=get_head_choices())
            
            refresh_heads_btn.click(fn=refresh_heads, outputs=head_selector)
            run_btn.click(fn=run_inference, inputs=head_selector, outputs=inference_results)
        
        with gr.Tab("📊 Performance"):
            gr.Markdown("## Memory & Performance Analysis")
            
            with gr.Row():
                with gr.Column():
                    memory_btn = gr.Button("Get Memory Usage", variant="secondary")
                    memory_info = gr.Markdown("")
                
                with gr.Column():
                    perf_btn = gr.Button("Compare Performance", variant="secondary")
                    perf_info = gr.Markdown("")
            
            memory_btn.click(fn=get_memory_info, outputs=memory_info)
            perf_btn.click(fn=compare_performance, outputs=perf_info)
        
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About Separable Weights
            
            The separable weights feature allows you to:
            
            ### 1. **Save/Load Backbone and Heads Separately**
            - Save backbone weights independently of head weights
            - Load pre-trained backbones with new heads
            - Share backbones across different tasks
            
            ### 2. **Feature Caching**
            - Extract backbone features once
            - Reuse features with multiple heads
            - Save computation when running multiple heads
            
            ### 3. **Dynamic Head Operations**
            - Load and unload heads dynamically
            - Add new heads to existing models
            - Freeze backbone/heads for transfer learning
            
            ### Benefits
            - **Memory Efficiency**: Only load needed heads
            - **Performance**: Cache features, run heads separately
            - **Flexibility**: Extend models without retraining backbone
            - **Modularity**: Mix and match backbones and heads
            
            ---
            
            **Project:** Multi-Task MobileNet V3 TFLite
            """)
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=False)


