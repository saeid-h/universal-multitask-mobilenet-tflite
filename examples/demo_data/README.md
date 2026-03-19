# Demo Data

This directory contains sample data for the separable weights demos.

## Directory Structure

```
demo_data/
├── sample_images/     # Sample images for inference demos
│   └── (images go here)
├── weights/           # Pre-trained weights for demos (optional)
│   ├── backbone_weights.h5
│   └── *_weights.h5
└── README.md          # This file
```

## Adding Sample Images

To use the demos with real images, add your images to the `sample_images/` directory:

1. Supported formats: JPEG, PNG
2. Recommended sizes: 96x96, 128x128, or 224x224 pixels
3. RGB (3 channels) or grayscale (1 channel) images

## Generating Synthetic Data

The demos can work with randomly generated data for testing purposes. No real images are required for basic functionality testing.

## Pre-trained Weights

To use pre-trained weights with the demos:

1. Create a model with `separable_weights=True`
2. Train the model
3. Save weights using `architecture.save_all_weights_separately('./demo_data/weights')`
4. Load in demos using `architecture.load_all_weights_separately('./demo_data/weights')`

## Example Usage

```python
from pathlib import Path
import numpy as np
from PIL import Image

# Load sample images
sample_dir = Path('demo_data/sample_images')
images = []
for img_path in sample_dir.glob('*.jpg'):
    img = Image.open(img_path).resize((128, 128))
    images.append(np.array(img))

# Stack into batch
batch = np.stack(images).astype(np.float32) / 255.0
```


