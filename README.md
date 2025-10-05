# Neural Style Transfer with PyTorch

This project implements Neural Style Transfer using PyTorch and a pre-trained VGG19 model. It blends a content image and a style image to create a stylized output image.

## Folder Structure
```
neural-style-transfer/
├── neural_style_transfer.py
├── README.md
├── images/
│   ├── content.jpg
│   └── style.jpg
└── output_image.jpg
```

- Place your content and style images in the `images/` folder and name them `content.jpg` and `style.jpg` (or update the script to use your filenames).

## Setup & Installation
1. **Clone the repository** (if not already):
   ```bash
   git clone <repo-url>
   cd neural-style-transfer
   ```
2. **Create a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install torch torchvision pillow
   ```

## Usage
1. Place your content and style images in the `images/` folder.
2. Run the script:
   ```bash
   python neural_style_transfer.py
   ```
3. The stylized image will be saved as `output_image.jpg` in the project root.

## GPU Acceleration
- The script will use GPU if available. Otherwise, it will run on CPU.

## References
- [PyTorch Neural Style Transfer Tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
