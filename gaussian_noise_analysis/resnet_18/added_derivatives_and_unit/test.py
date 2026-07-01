import train
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

def test(model, test_data, device):
    """
    Test the model on the provided test data.

    Args:
        model: The trained model to be tested.
        test_data: The data to test the model on.

    Returns:
        accuracy: The accuracy of the model on the test data.
    """
    # Assuming train has a function to evaluate the model
    print("Evaluating on Clean Validation Set...")
    clean_total = 0
    clean_correct = 0
    with torch.no_grad():
        for images, labels in test_data:
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(model(images).data, 1)
            clean_total += labels.size(0)
            clean_correct += (predicted == labels).sum().item()
    clean_acc = 100 * clean_correct / clean_total
    return clean_acc

if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    higher_order_of_noise_val_transforms = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomApply([train.AddGaussianNoise(std=2.0)], p=1.0)
    ])
    train_dir, val_dir = train.download_and_extract_imagenette()
    higher_order_of_noise_val_dataset = train.datasets.ImageFolder(val_dir, transform=higher_order_of_noise_val_transforms, target_transform=train.map_class_to_imagenet)
    higher_order_of_noise_val_loader = torch.utils.data.DataLoader(higher_order_of_noise_val_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    print("Loading Pretrained ResNet-18...")
    base_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
    
    print("Wrapping with U-Net Edge-Aware Robustifier...")
    model = train.EdgeAwareRobustifier(
        base_resnet=base_resnet,
        blur_kernel=7,
        blur_sigma=1,
        unet_in=5,
        unet_out=3
    ).to(device)
    model.load_state_dict(torch.load("kernel7_noise1.pth", map_location=device))
    
    sum_test_loss = 0
    sum_accuracy = 0
    for i in range(5):
        higher_order_acc = test(model, higher_order_of_noise_val_loader, device)
        sum_accuracy += higher_order_acc
        print(f"Round {i+1}: Higher Order Accuracy: {higher_order_acc:.2f}%")

    print(f"Average Higher Order Accuracy: {sum_accuracy/5:.2f}%")

