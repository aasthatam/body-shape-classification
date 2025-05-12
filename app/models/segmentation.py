import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

class SegmentationModel:
    def __init__(self, weights_path=None):
        """
        Initialize the DeepLabV3 model for person segmentation.
        
        Args:
            weights_path: Path to pretrained weights file or None to use pretrained=True
        """
        # Initialize model with pretrained weights
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use the newer import method with weights enum for recent PyTorch versions
        try:
            # For PyTorch 1.13+ and torchvision 0.14+
            self.model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
        except (TypeError, AttributeError):
            # Fallback for older PyTorch versions
            self.model = deeplabv3_resnet101(pretrained=True)
            
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Load custom weights if provided
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Class index for 'person' in COCO dataset
        self.person_class = 15

    def segment_person(self, image):
        """
        Remove background from an image, isolating the person.
        
        Args:
            image: OpenCV/NumPy image in BGR format
            
        Returns:
            Segmented image with background removed (person only)
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb_image.shape[:2]
        
        # Prepare image for model
        input_tensor = self.transform(rgb_image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        
        # Get mask for the person class
        person_mask = output.argmax(0).byte().cpu().numpy() == self.person_class
        
        # Apply mask to original image
        mask = person_mask.astype(np.uint8) * 255
        mask_3channel = cv2.merge([mask, mask, mask])
        
        # Create segmented image
        segmented_image = cv2.bitwise_and(image, mask_3channel)
        
        return segmented_image, mask 