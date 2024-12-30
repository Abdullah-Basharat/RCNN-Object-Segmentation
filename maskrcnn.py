import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw


def load_model():
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model


def predict_image(model, image_path, threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    transformed_image = F.to_tensor(image)

    with torch.no_grad():
        outputs = model([transformed_image])

    scores = outputs[0]['scores']
    indices = [i for i, score in enumerate(scores) if score > threshold]
    filtered_boxes = outputs[0]['boxes'][indices]
    filtered_masks = outputs[0]['masks'][indices]


    return image, filtered_boxes, filtered_masks


def visualize_results(image, boxes, masks):
    draw = ImageDraw.Draw(image)

    for i in range(len(boxes)):
        draw.rectangle(boxes[i].tolist(), outline="red", width=5)


    for mask in masks:
        mask_np = mask[0].mul(255).byte().cpu().numpy()
        mask_img = Image.fromarray(mask_np).resize(image.size, resample=Image.BILINEAR)
        image = Image.blend(image, mask_img.convert("RGB"), alpha=0.7)

    return image
