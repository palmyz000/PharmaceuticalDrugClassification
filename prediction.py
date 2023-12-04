from typing import List, Tuple
import torch
import torchvision.transforms as T
from PIL import Image

def pred_class(model: torch.nn.Module,
               image,
               class_names: List[str],
               image_size: Tuple[int, int] = (224, 224),
               ):
    # Check device availability
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Move model and input image to the selected device
    model.to(device)

    # Create image transformation
    image_transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Open image
    img = image

    # Transform image and send it to the target device
    transformed_image = image_transform(img).unsqueeze(dim=0).float()
    transformed_image = transformed_image.to(device)

    # Set the model to evaluation mode and make predictions
    model.eval()
    with torch.inference_mode():
        target_image_pred = model(transformed_image)

    # Convert logits to prediction probabilities
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert prediction probabilities to labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    classname = class_names[target_image_pred_label]
    prob = target_image_pred_probs.cpu().numpy()

    return prob
