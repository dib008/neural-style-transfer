import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import copy
import os

# Device configuration: use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image loading and preprocessing
def load_image(img_path, max_size=512, shape=None):
    image = Image.open(img_path).convert('RGB')
    size = max_size if max(image.size) > max_size else max(image.size)
    if shape:
        size = shape
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3, :, :]),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image.to(device)

# Load content and style images
content_img_path = os.path.join('images', 'content.jpg')
style_img_path = os.path.join('images', 'style.jpg')
output_img_path = 'output_image.jpg'

content = load_image(content_img_path)
style = load_image(style_img_path, shape=content.shape[-2:])

# Define the VGG model for feature extraction
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=True).features[:29]
    def forward(self, x):
        features = []
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.chosen_features:
                features.append(x)
        return features

# Gram matrix for style representation
def gram_matrix(tensor):
    _, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Initialize generated image
generated = content.clone().requires_grad_(True)

# Hyperparameters
num_steps = 300
style_weight = 1e6
content_weight = 1

# Model and optimizer
model = VGG().to(device).eval()
optimizer = optim.Adam([generated], lr=0.003)

for step in range(num_steps):
    generated_features = model(generated)
    content_features = model(content)
    style_features = model(style)

    content_loss = torch.mean((generated_features[2] - content_features[2]) ** 2)
    style_loss = 0
    for gen_feat, style_feat in zip(generated_features, style_features):
        G = gram_matrix(gen_feat)
        A = gram_matrix(style_feat)
        style_loss += torch.mean((G - A) ** 2)
    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step [{step}/{num_steps}], Content Loss: {content_loss.item():.4f}, Style Loss: {style_loss.item():.4f}")

# Postprocessing and saving the output image
def save_image(tensor, path):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)
    print(f"Output image saved as {path}")

save_image(generated, output_img_path)

# --- End of script ---
# For more details, see README.md
