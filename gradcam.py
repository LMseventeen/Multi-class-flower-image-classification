import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models import get_modified_resnet18
from torchvision import transforms
from PIL import Image

def preprocess_image(img_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)
    return img, img_tensor

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def __call__(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        loss = output[0, class_idx]
        self.model.zero_grad()
        loss.backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

def show_cam_on_image(img, mask, alpha=0.5):
    # 如果是PIL.Image，先转为numpy数组
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    if img.max() > 1.0:
        img = img / 255.0
    cam = heatmap / 255.0 * alpha + img * (1 - alpha)
    cam = np.clip(cam, 0, 1)
    plt.imshow(cam)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    import sys
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_modified_resnet18().to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    # 选择一张图片路径
    img_path = sys.argv[1] if len(sys.argv) > 1 else './data/flowers17_imgfolder/0/image_0001.jpg'
    img, img_tensor = preprocess_image(img_path)
    img_tensor = img_tensor.to(device)
    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)
    cam_mask = gradcam(img_tensor)
    show_cam_on_image(img, cam_mask)
    gradcam.remove_hooks() 