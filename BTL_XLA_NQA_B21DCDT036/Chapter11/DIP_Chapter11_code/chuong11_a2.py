import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import nn
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Tải ảnh content và style
content_image_path = r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\monalisa.jpg"
style_image_path = r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\starry_night.jpg"


# Chuyển ảnh thành Tensor và chuẩn hóa
def image_loader(image_name, imsize=(512, 512)):
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image


content_image = image_loader(content_image_path)
style_image = image_loader(style_image_path)


# Hiển thị ảnh content và style
def imshow(tensor):
    image = tensor.cpu().clone()  # clone the tensor to avoid changing the original tensor
    image = image.squeeze(0)  # remove the batch dimension
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


# Hiển thị ảnh
imshow(content_image)
imshow(style_image)

# Load pre-trained VGG19 model
cnn = models.vgg19(pretrained=True).features

# Chuyển model sang chế độ evaluation
for param in cnn.parameters():
    param.requires_grad = False

# Di chuyển model và ảnh vào GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = cnn.to(device)
content_image = content_image.to(device)
style_image = style_image.to(device)


# Define các loss layers và các trọng số cho các loss (content và style loss)
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, x):
        loss = nn.functional.mse_loss(x, self.target)
        return loss


class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target).detach()

    def gram_matrix(self, x):
        batch_size, h, w, f_map_num = x.size()
        features = x.view(batch_size, f_map_num, h * w)
        G = torch.bmm(features, features.transpose(1, 2))
        return G.div(f_map_num * h * w)

    def forward(self, x):
        G = self.gram_matrix(x)
        loss = nn.functional.mse_loss(G, self.target)
        return loss


# Tạo model với các lớp chứa content loss và style loss
def get_style_model_and_losses(cnn, content_img, style_img):
    cnn = cnn.clone()

    content_losses = []
    style_losses = []

    # Thêm content loss và style loss vào các lớp
    model = nn.Sequential()
    i = 0
    for layer in cnn.children():
        model.add_module(str(i), layer)
        if isinstance(layer, nn.Conv2d):
            i += 1
            if i == 4:
                content_loss = ContentLoss(content_img)
                model.add_module(str(i), content_loss)
                content_losses.append(content_loss)
            if i == 2:
                style_loss = StyleLoss(style_img)
                model.add_module(str(i), style_loss)
                style_losses.append(style_loss)
    return model, content_losses, style_losses


# Lưu model
model, content_losses, style_losses = get_style_model_and_losses(cnn, content_image, style_image)

# Cập nhật các tham số
input_img = content_image.clone()

# Tối ưu hóa input image
optimizer = optim.LBFGS(input_img)

# Chu trình huấn luyện
run = [0]
while run[0] <= 300:
    def closure():
        input_img.data.clamp_(0, 1)

        optimizer.zero_grad()
        model(input_img)

        style_score = 0
        content_score = 0

        for sl in style_losses:
            style_score += sl.forward(input_img)
        for cl in content_losses:
            content_score += cl.forward(input_img)

        loss = style_score * 1000000 + content_score * 1
        loss.backward()

        run[0] += 1
        return loss


    optimizer.step(closure)

# Hiển thị kết quả
imshow(input_img)