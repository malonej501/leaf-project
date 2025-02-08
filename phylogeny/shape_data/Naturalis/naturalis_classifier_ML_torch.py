import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import random
import shutil
import pandas as pd
from PIL import Image

prop_train = 0.8  # define proportion of data used for training

wd = "Naturalis_eud_sample_Janssens_intersect_21-01-24_ML-data"

label_data = pd.read_csv(
    "sample_eud_21-1-24/img_labels_unambig_full.csv",
    names=["species", "shape"],
    sep="\t",
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transform for preprocessing images
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Load data
def split_test_train():
    all_files = os.listdir(wd)
    all_imgs = [file for file in all_files if file.endswith(".png")]
    random.shuffle(all_imgs)
    split_index = int(len(all_imgs) * prop_train)
    train_imgs = all_imgs[:split_index]
    test_imgs = all_imgs[split_index:]
    # os.makedirs("Naturalis_eud_sample_Janssens_intersect_21-01-24_ML-data/train")
    # os.makedirs("Naturalis_eud_sample_Janssens_intersect_21-01-24_ML-data/test")
    for file in train_imgs:
        shutil.move(
            f"{wd}/{file}",
            f"{wd}/train/{file}",
        )
    for file in test_imgs:
        shutil.move(
            f"{wd}/{file}",
            f"{wd}/test/{file}",
        )


def label_imgs():
    for dir in os.listdir(wd):
        # for dir in dirs:
        #     for shape in range(0, 4):
        #         os.makedirs(os.path.join(root, dir, str(shape)))
        for file in os.listdir(os.path.join(wd, dir)):
            if file.endswith(".png"):
                print(file)
                row = label_data[
                    label_data["species"].apply(lambda substr: substr in file)
                ]
                if not row.empty:
                    label = row.iloc[0, 1]
                    img_path = os.path.join(
                        wd,
                        dir,
                        file,
                    )

                    shutil.move(img_path, f"{wd}/{dir}/{str(label)}/{file}")


def initialise_data():
    print("Initialising data...")
    train_dataset = datasets.ImageFolder(
        f"{wd}/train/",
        transform=transform,
    )
    test_dataset = datasets.ImageFolder(
        f"{wd}/test/",
        transform=transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader


def initialise_model(train_dataset):
    print("Initialising model...")
    # Load pre-trained ResNet model
    model = models.resnet18(pretrained=True)

    # Freeze parameters so we don't backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    # Modify the last fully connected layer for the number of classes in your dataset
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))

    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    return model, optimizer, criterion


def train(train_dataset, train_loader, model, optimizer, criterion):
    print("Beginning training...")
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    return model


def evalutate(model, test_loader):
    print("Testing model...")
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), f"{wd}/torch_model.pth")


def train_and_test():
    train_dataset, test_dataset, train_loader, test_loader = initialise_data()
    model, optimizer, criterion = initialise_model(train_dataset)
    model = train(train_dataset, train_loader, model, optimizer, criterion)
    evalutate(model, test_loader)


def predict1():
    # train_and_test()
    train_dataset, test_dataset, train_loader, test_loader = initialise_data()
    model, optimizer, criterion = initialise_model(train_dataset)
    model.load_state_dict(torch.load(f"{wd}/torch_model.pth"))

    results_df = pd.DataFrame(columns=["Filename", "Predicted_Class", "Actual_Class"])
    for category in os.listdir(f"{wd}/test"):
        for item in os.listdir(f"{wd}/test/{category}"):
            if item.endswith(".png"):
                img = Image.open(f"{wd}/test/{category}/{item}")
                img = transform(img)
                img = img.unsqueeze(0)

                # Perform inference
                with torch.no_grad():
                    outputs = model(img)
                    print(outputs)
                    _, predicted = torch.max(outputs, 1)
                    predicted_class = predicted.item()

                # Map predicted class index to class name (replace with your actual class names)
                # Example: class_names = ['class1', 'class2', ...]
                # class_names = ["class1", "class2", ...]
                # predicted_class_name = class_names[predicted_class]

                actual_class = category

                # Append the result to the DataFrame
                results_df.loc[len(results_df)] = [item, predicted_class, actual_class]

    print(results_df)
    results_df.to_csv(f"{wd}/torch_model_prediction.csv", index=False)


if __name__ == "__main__":
    classes = [0, 1, 2, 3]
    train_dataset, test_dataset, train_loader, test_loader = initialise_data()
    model, optimizer, criterion = initialise_model(train_dataset)
    model.load_state_dict(torch.load(f"{wd}/torch_model.pth"))

    results_df = pd.DataFrame(columns=["Filename", "Predicted_Class", "Actual_Class"])
    model.eval()
    for i in range(len(test_dataset)):
        filename = test_dataset.imgs[i][0]
        x, y = test_dataset[i][0].unsqueeze(0), test_dataset[i][1]
        with torch.no_grad():
            x = x.to(device)
            pred = model(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            print(f"Predicted: {predicted}, Actual: {actual},\t\tFilename: {filename}")
            results_df.loc[len(results_df)] = [filename, predicted, actual]

    results_df.to_csv(f"{wd}/torch_model_prediction.csv", index=False)
