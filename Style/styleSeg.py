import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from torch.nn.functional import interpolate
import numpy as np
import PIL

from style import run_style_transfer, image_loader, imshow
from pspnet import PSPNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def shortside_resize(image, mask=None, size=256):
    h, w = image.shape[:2]
    if h >= w:
        new_w = size
        new_h = int(h * (size*1.0/w))
        if new_h % 32 != 0:
            new_h = new_h + 32 - new_h % 32
    else:
        new_h = size
        new_w = int(w * (size*1.0/h))
        if new_w % 32 != 0:
            new_w = new_w + 32 - new_w % 32
    new_size = (int(new_w), int(new_h))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    if mask is not None:
        resized_mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
        return resized_image, resized_mask
    else:
        return resized_image


def get_mask(small_mask, ratio=2):  # get label mask
    small_mask = (small_mask > 0).astype(np.uint8)
    small_mask = torch.tensor(small_mask)
    updim_small_mask = small_mask.unsqueeze(0).unsqueeze(0)  # dim2 to dim4
    updim_big_mask = interpolate(updim_small_mask, scale_factor=ratio)
    big_mask = updim_big_mask.squeeze(0).squeeze(0)  # dim4 to dim2
    return big_mask

unloader = transforms.ToPILImage()


def imsave(tensor, name):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    print(type(image))
    image.save(name)


def get_style_seg(image, style, mask):
    # the tensor shape is (1, 3 ,size, size),size is usually 512
    image = image.cpu().clone()
    image = image.squeeze(0)
    style = style.cpu().clone()
    style = style.squeeze(0)
    background_mask = torch.ones(*mask.shape) - mask
    style_seg = mask*image + background_mask*style
    return style_seg



# desired size of the output image
imgsize = 512 if torch.cuda.is_available() else 128

loader = transforms.Compose([        
    transforms.Resize([imgsize, imgsize]),       
    transforms.ToTensor()])          


seg_path = './seg/epoch_50_iou0.81.pth'
stylepath = './img_style'
contentpath = './img_content'
savepath = './save_style_seg'

style_name = stylepath + r'/style8.png'
content_name = contentpath + r'/tryx.jpg'
save_name = savepath + r'/tryx_3.jpg'
 
style_img = image_loader(style_name)      
content_img = image_loader(content_name)    

print(style_img.shape)
print(content_img.shape)
assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"


cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

input_img = content_img.clone() 

style = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

print(style.shape)

model = PSPNet(n_classes=6)  # seg model
model.load_state_dict(torch.load(seg_path))
model.eval()

# seg part
image_np = cv2.imread(content_name)
image_np = shortside_resize(image_np)

image = image_np / 127.5 - 1
image = np.transpose(image, (2, 0, 1))
image_th = torch.from_numpy(image).float().unsqueeze(0)
with torch.no_grad():
    seg_logit = model(image_th)
seg_pred = torch.argmax(seg_logit, dim=1)[0].cpu().numpy()
seg_pred = seg_pred.astype(np.uint8)

final_seg = get_mask(seg_pred)

'''
print('final_seg', final_seg, final_seg.shape)
print('background_seg', background_seg, background_seg.shape)
print(background_seg[230:232, :])
'''
# print(content_img.shape)

final_result = get_style_seg(content_img, style, final_seg)
final_result = final_result.unsqueeze(0)
imsave(final_result, save_name)




