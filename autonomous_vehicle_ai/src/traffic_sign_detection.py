"""
Traffic Sign Detection and Classification.

Uses:
- MobileNetV3-Small transfer learning
- GTSRB 43-class traffic sign dataset
- Apple MPS when available
- Existing YOLO ObjectDetector for stop-sign detection
"""

import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V3_Small_Weights


class TrafficSignClassifier:
    """Deep-learning traffic sign classifier using MobileNetV3-Small."""

    CLASS_NAMES = [
        "Speed limit 20 km/h",
        "Speed limit 30 km/h",
        "Speed limit 50 km/h",
        "Speed limit 60 km/h",
        "Speed limit 70 km/h",
        "Speed limit 80 km/h",
        "End of speed limit 80 km/h",
        "Speed limit 100 km/h",
        "Speed limit 120 km/h",
        "No passing",
        "No passing for vehicles over 3.5 tons",
        "Right-of-way at next intersection",
        "Priority road",
        "Yield",
        "Stop",
        "No vehicles",
        "Vehicles over 3.5 tons prohibited",
        "No entry",
        "General caution",
        "Dangerous curve left",
        "Dangerous curve right",
        "Double curve",
        "Bumpy road",
        "Slippery road",
        "Road narrows on the right",
        "Road work",
        "Traffic signals",
        "Pedestrians",
        "Children crossing",
        "Bicycles crossing",
        "Beware of ice/snow",
        "Wild animals crossing",
        "End of all speed and passing limits",
        "Turn right ahead",
        "Turn left ahead",
        "Ahead only",
        "Go straight or right",
        "Go straight or left",
        "Keep right",
        "Keep left",
        "Roundabout mandatory",
        "End of no passing",
        "End of no passing by vehicles over 3.5 tons",
    ]

    NUM_CLASSES = 43

    def __init__(
        self,
        num_classes=43,
        learning_rate=0.001,
        batch_size=32,
        epochs=5,
        model_path="models/traffic_sign_mobilenet.pth",
        device=None,
    ):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_path = model_path

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = torch.device(device)

        self.model = None

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def prepare_model(self, pretrained=True):
        """Create MobileNetV3-Small and configure it for GTSRB."""
        if pretrained:
            weights = MobileNet_V3_Small_Weights.DEFAULT
        else:
            weights = None

        self.model = models.mobilenet_v3_small(weights=weights)

        # Freeze the pretrained feature extractor.
        for parameter in self.model.features.parameters():
            parameter.requires_grad = False

        # Replace the final classifier for 43 GTSRB classes.
        input_features = self.model.classifier[3].in_features

        self.model.classifier[3] = nn.Linear(
            input_features,
            self.num_classes,
        )

        self.model = self.model.to(self.device)

        return self.model

    def train(
        self,
        data_dir="dataset/gtsrb",
        validation_split=0.2,
        epochs=None,
        batch_size=None,
        learning_rate=None,
        save_best=True,
    ):
        """
        Train the classifier using the GTSRB dataset.

        The dataset is downloaded only when this method is explicitly called.
        """
        if self.model is None:
            self.prepare_model(pretrained=True)

        epochs = epochs or self.epochs
        batch_size = batch_size or self.batch_size
        learning_rate = learning_rate or self.learning_rate

        os.makedirs(data_dir, exist_ok=True)

        print("Loading GTSRB dataset...")
        print(f"Device: {self.device}")

        full_dataset = datasets.GTSRB(
            root=data_dir,
            split="train",
            download=True,
            transform=self.transform,
        )

        validation_size = int(
            len(full_dataset) * validation_split
        )

        training_size = len(full_dataset) - validation_size

        train_dataset, validation_dataset = random_split(
            full_dataset,
            [training_size, validation_size],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            self.model.classifier[3].parameters(),
            lr=learning_rate,
        )

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        best_val_accuracy = 0.0

        print(f"Training samples: {training_size}")
        print(f"Validation samples: {validation_size}")
        print(f"Epochs: {epochs}")

        for epoch in range(epochs):
            epoch_start = time.time()

            self.model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)

                predictions = outputs.argmax(dim=1)
                correct += (
                    predictions == labels
                ).sum().item()

                total += labels.size(0)

            train_loss = running_loss / total
            train_accuracy = correct / total

            # Validation.
            self.model.eval()

            val_loss_total = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in validation_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                    val_loss_total += (
                        loss.item() * images.size(0)
                    )

                    predictions = outputs.argmax(dim=1)

                    val_correct += (
                        predictions == labels
                    ).sum().item()

                    val_total += labels.size(0)

            val_loss = val_loss_total / val_total
            val_accuracy = val_correct / val_total

            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)

            elapsed = time.time() - epoch_start

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy * 100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy * 100:.2f}% | "
                f"Time: {elapsed:.1f}s"
            )

            if save_best and val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_model(self.model_path)
                print(
                    f"Best model saved: {self.model_path}"
                )

        return history

    def save_model(self, path=None):
        """Save model weights and configuration."""
        if self.model is None:
            raise RuntimeError(
                "Model has not been prepared."
            )

        path = path or self.model_path

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "num_classes": self.num_classes,
                "class_names": self.CLASS_NAMES,
            },
            path,
        )

        return path

    def load_model(self, path=None):
        """Load trained model weights."""
        path = path or self.model_path

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found: {path}"
            )

        if self.model is None:
            self.prepare_model(pretrained=False)

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        return self.model

    def predict(self, image):
        """
        Predict a traffic sign from an OpenCV BGR image.

        Returns:
            class_id
            class_name
            confidence
        """
        if self.model is None:
            raise RuntimeError(
                "Model is not prepared or loaded."
            )

        if image is None:
            raise ValueError("Input image is None.")

        self.model.eval()

        # OpenCV uses BGR; torchvision models expect RGB.
        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        from PIL import Image

        pil_image = Image.fromarray(rgb)

        tensor = self.transform(pil_image)
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1,
            )

        class_id = int(prediction.item())
        confidence_value = float(
            confidence.item()
        )

        return {
            "class_id": class_id,
            "class_name": self.CLASS_NAMES[class_id],
            "confidence": confidence_value,
        }

    def predict_image(self, image_path):
        """Load an image from disk and classify the traffic sign."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(
                f"Could not read image: {image_path}"
            )

        return self.predict(image)

    def draw_prediction(
        self,
        image,
        prediction,
        position=(20, 40),
    ):
        """Draw a traffic-sign prediction on an image."""
        output = image.copy()

        label = (
            f"{prediction['class_name']} "
            f"({prediction['confidence'] * 100:.1f}%)"
        )

        cv2.putText(
            output,
            label,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return output

    def detect_and_classify_with_yolo(
        self,
        image,
        object_detector,
        confidence_threshold=0.25,
    ):
        """
        Detect stop signs using the existing YOLO detector,
        then classify detected sign crops.
        """
        if image is None:
            raise ValueError("Input image is None.")

        detections, _ = object_detector.detect(image)

        results = []

        for detection in detections:
            class_name = detection["class_name"]

            if class_name.lower() != "stop sign":
                continue

            if detection["confidence"] < confidence_threshold:
                continue

            x1, y1, x2, y2 = detection["box"]

            height, width = image.shape[:2]

            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(width, int(x2))
            y2 = min(height, int(y2))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            try:
                prediction = self.predict(crop)
            except RuntimeError:
                prediction = {
                    "class_id": -1,
                    "class_name": "Classifier not loaded",
                    "confidence": 0.0,
                }

            enriched_detection = dict(detection)
            enriched_detection["traffic_sign"] = prediction

            results.append(enriched_detection)

        return results

    def annotate_detected_signs(
        self,
        image,
        detections,
    ):
        """Draw YOLO boxes and traffic-sign classification results."""
        output = image.copy()

        for detection in detections:
            x1, y1, x2, y2 = map(
                int,
                detection["box"],
            )

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                3,
            )

            sign = detection.get(
                "traffic_sign",
                {},
            )

            class_name = sign.get(
                "class_name",
                detection["class_name"],
            )

            confidence = sign.get(
                "confidence",
                detection["confidence"],
            )

            label = (
                f"{class_name} "
                f"{confidence * 100:.1f}%"
            )

            cv2.putText(
                output,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return output


if __name__ == "__main__":
    print(
        "Traffic sign classification module loaded successfully."
    )
