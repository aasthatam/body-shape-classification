import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights

class PoseEstimationModel:
    def __init__(self, weights_path=None):
        """
        Initialize the model for human pose estimation.
        
        Args:
            weights_path: Path to pretrained weights file or None to use pretrained=True
        """
        # Initialize model with pretrained weights
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use the newer import method with weights enum for recent PyTorch versions
        try:
            # For PyTorch 1.13+ and torchvision 0.14+
            self.model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
        except (TypeError, AttributeError):
            # Fallback for older PyTorch versions
            self.model = keypointrcnn_resnet50_fpn(pretrained=True)
            
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Load custom weights if provided
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # COCO keypoints and their connections for visualization
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Key indices we'll be using for body shape analysis
        self.key_indices = {
            'left_shoulder': 5,
            'right_shoulder': 6,
            'left_hip': 11,
            'right_hip': 12,
            'left_knee': 13,
            'right_knee': 14,
            'left_ankle': 15,
            'right_ankle': 16
        }

    def detect_keypoints(self, image):
        """
        Detect human pose keypoints in an image.
        
        Args:
            image: OpenCV/NumPy image in BGR format
            
        Returns:
            Dictionary of keypoints with coordinates
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prepare image for model
        input_tensor = self.transform(rgb_image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_batch)
        
        # Get the keypoints if a person is detected
        if len(output[0]['keypoints']) > 0:
            # Use the first person with highest score
            keypoints = output[0]['keypoints'][0].cpu().numpy()
            scores = output[0]['keypoints_scores'][0].cpu().numpy()
            
            # Create keypoints dictionary
            keypoints_dict = {}
            for i, (name, kp, score) in enumerate(zip(self.keypoint_names, keypoints, scores)):
                if score > 0.5:  # Filter low confidence keypoints
                    keypoints_dict[name] = (int(kp[0]), int(kp[1]))
                else:
                    keypoints_dict[name] = None
            
            return keypoints_dict
        else:
            return None
    
    def visualize_keypoints(self, image, keypoints):
        """
        Draw keypoints and connections on the image for visualization.
        
        Args:
            image: OpenCV/NumPy image in BGR format
            keypoints: Dictionary of keypoints with coordinates
            
        Returns:
            Image with keypoints drawn
        """
        vis_img = image.copy()
        
        # Define connections for visualization
        connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_shoulder', 'left_elbow'),
            ('right_shoulder', 'right_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_elbow', 'right_wrist'),
            ('left_hip', 'left_knee'),
            ('right_hip', 'right_knee'),
            ('left_knee', 'left_ankle'),
            ('right_knee', 'right_ankle')
        ]
        
        # Draw keypoints
        for name, point in keypoints.items():
            if point:
                cv2.circle(vis_img, point, 5, (0, 255, 0), -1)
        
        # Draw connections
        for start_name, end_name in connections:
            start_point = keypoints.get(start_name)
            end_point = keypoints.get(end_name)
            
            if start_point and end_point:
                cv2.line(vis_img, start_point, end_point, (0, 0, 255), 2)
        
        return vis_img 