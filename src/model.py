from torch import nn
import torchvision.models as models


class Net(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pooling = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(128 * 16 * 16, 128)
        self.output = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pooling(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pooling(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pooling(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(self.linear(x))
        x = self.output(x)

        return x


def create_model(
    model_name: str, num_classes: int, pretrained: bool = True, dropout: float = 0.5
):
    if model_name.lower() == "custom":
        return Net(num_classes, dropout)

    if model_name.lower() == "resnet18":
        if pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet18(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes)
        )
        return model

    if model_name.lower() == "resnet34":
        if pretrained:
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet34(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes)
        )
        return model

    if model_name.lower() == "efficientnet_b0":
        if pretrained:
            model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
        else:
            model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(model.classifier[1].in_features, num_classes)
        )
        return model

    raise ValueError(
        f"Unknown model: {model_name}. Options: custom, resnet18, resnet34, efficientnet_b0"
    )


def freeze_backbone(model, model_name: str):
    if model_name.lower() == "custom":
        return

    if "resnet" in model_name.lower():
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    elif "efficientnet" in model_name.lower():
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False


def unfreeze_backbone(model, model_name: str):
    if model_name.lower() == "custom":
        return

    for param in model.parameters():
        param.requires_grad = True


def get_feature_extractor(model, model_name: str):
    if "resnet" in model_name.lower():
        return nn.Sequential(*list(model.children())[:-1])
    elif "efficientnet" in model_name.lower():
        return nn.Sequential(model.features, model.avgpool)
    return None


def get_target_layer(model, model_name: str):
    if model_name.lower() == "custom":
        return model.conv3

    if "resnet" in model_name.lower():
        return model.layer4[-1]

    if "efficientnet" in model_name.lower():
        return model.features[-1]

    return None
