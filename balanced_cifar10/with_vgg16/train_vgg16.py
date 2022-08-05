import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

from copy import deepcopy
from datetime import datetime


# -------------------- Parameter -------------------- #

SEED = 42
torch.manual_seed(SEED)

USE_CUDA = True

N_EPOCHS = 10
BATCH_SIZE = 32
TRAIN_VALIDATION_SPLIT = 0.9
LOG_ON_DIFF_TOL = 0.01

MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
LEARN_RATE = 0.001
LEARN_RATE_STEP_SIZE = 2
LEARN_RATE_GAMMA = 0.9

IMAGE_TRANSFORMATIONS = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                 (0.2023, 0.1994, 0.2010))])


# ----------------------- Utils ---------------------- #

if USE_CUDA and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ----------------------- Steps ---------------------- #

def make_folder():
    time_stamp = str(datetime.now().replace(microsecond=0))
    time_stamp = time_stamp.replace(":", "-").replace(" ", "_")
    folder_path = os.path.join(os.getcwd(), time_stamp)
    os.mkdir(folder_path)
    return folder_path


def load_CIFAR10(batch_size: int) -> (DataLoader, DataLoader, DataLoader):
    training_dataset: Dataset = torchvision.datasets.CIFAR10(root='../../dataset',
                                                             train=True,
                                                             download=True,
                                                             transform=IMAGE_TRANSFORMATIONS)

    test_dataset: Dataset = torchvision.datasets.CIFAR10(root='../../dataset',
                                                         train=False,
                                                         download=True,
                                                         transform=IMAGE_TRANSFORMATIONS)

    sample_size = len(training_dataset)
    training_dataset_size = int(sample_size * TRAIN_VALIDATION_SPLIT)
    validation_dataset_size = sample_size - training_dataset_size

    training_dataset, validation_dataset = torch.utils.data. \
        random_split(training_dataset, lengths=(training_dataset_size, validation_dataset_size))

    training_data_loader: DataLoader = torch.utils.data.DataLoader(training_dataset,
                                                                   batch_size=batch_size,
                                                                   pin_memory=True,
                                                                   shuffle=True)

    validation_data_loader: DataLoader = torch.utils.data.DataLoader(validation_dataset,
                                                                     batch_size=batch_size,
                                                                     pin_memory=True,
                                                                     shuffle=True)

    test_data_loader: DataLoader = torch.utils.data.DataLoader(test_dataset,
                                                               batch_size=batch_size,
                                                               pin_memory=True,
                                                               shuffle=False)

    return training_data_loader, validation_data_loader, test_data_loader


def prepare_model() -> nn.Module:
    model = models.vgg16(weights='VGG16_Weights.DEFAULT')

    # train the entire model
    for param in model.parameters():
        param.requires_grad = True

    model.avgpool = nn.Identity()
    model.classifier = nn.Sequential(nn.Linear(512, 128, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=0.25, inplace=False),
                                     nn.Linear(128, 32, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=0.25, inplace=False),
                                     nn.Linear(32, 10, bias=True)
                                     )
    model.to(DEVICE)
    return model


def fit(model: torch.nn.Module, optimizer: torch.optim.Optimizer, training_data: DataLoader) -> {str, float}:
    model.train()
    outputs = {"training_loss": [], "training_accuracy": []}
    for samples, labels in training_data:
        samples = samples.to(DEVICE)
        labels = labels.to(DEVICE)
        output = model(samples)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        _, prediction = torch.max(output, dim=1)
        accuracy = torch.tensor(torch.sum(prediction == labels).item() / len(prediction))

        outputs["training_loss"].append(loss)
        outputs["training_accuracy"].append(accuracy)

    avg_training_loss = torch.stack(outputs["training_loss"]).mean().item()
    avg_training_accuracy = torch.stack(outputs["training_accuracy"]).mean().item()

    return {
            "avg_training_loss": avg_training_loss,
            "avg_training_accuracy": avg_training_accuracy,
            }


def evaluate(model: torch.nn.Module, validation_data: DataLoader) -> {str, float}:
    model.eval()
    with torch.no_grad():
        outputs = {"validation_loss": [], "validation_accuracy": []}
        for samples, labels in validation_data:
            samples = samples.to(DEVICE)
            labels = labels.to(DEVICE)
            output = model(samples)
            loss = F.cross_entropy(output, labels)
            _, prediction = torch.max(output, dim=1)
            accuracy = torch.tensor(torch.sum(prediction == labels).item() / len(prediction))

            outputs["validation_loss"].append(loss)
            outputs["validation_accuracy"].append(accuracy)

        avg_validation_loss = torch.stack(outputs["validation_loss"]).mean().item()
        avg_validation_accuracy = torch.stack(outputs["validation_accuracy"]).mean().item()

        return {
                "avg_validation_loss": avg_validation_loss,
                "avg_validation_accuracy": avg_validation_accuracy,
                }


def train_model(model: nn.Module, optimizer: optim.Optimizer, scheduler: lr_scheduler,
                training_data: DataLoader, validation_data: DataLoader) -> (nn.Module, {str, float}):

    chosen_model_param = None
    chosen_model_acc = 0

    collected_scores = []
    for ep in range(N_EPOCHS):

        training_scores = fit(model=model, optimizer=optimizer, training_data=training_data)
        validation_scores = evaluate(model=model, validation_data=validation_data)
        scheduler.step()

        accuracy = validation_scores["avg_validation_accuracy"]
        if accuracy > chosen_model_acc + LOG_ON_DIFF_TOL:
            chosen_model_param = deepcopy(model.state_dict())
            chosen_model_acc = accuracy

        scores = {**training_scores, **validation_scores}
        print_scores(episode=ep, score_dict=scores)
        collected_scores.append(scores)

    model.load_state_dict(chosen_model_param)
    combined_scores = {k: [dic[k] for dic in collected_scores] for k in collected_scores[0]}

    return model, combined_scores


def test_model(model: torch.nn.Module, test_data: DataLoader) -> {str, float}:
    model.eval()
    collect_labels = []
    collect_predictions = []
    with torch.no_grad():
        for samples, labels in test_data:
            samples = samples.to(DEVICE)
            labels = labels.to(DEVICE)
            output = model(samples)
            _, prediction = torch.max(output, dim=1)
            collect_labels.append(labels)
            collect_predictions.append(prediction)

    collected_labels = torch.cat(collect_labels).cpu().numpy()
    collected_predictions = torch.cat(collect_predictions).cpu().numpy()

    roc_scores = calculate_test_scores(y_true=collected_labels,
                                       y_predict=collected_predictions,
                                       class_names=test_data.dataset.classes
                                       )
    return roc_scores


def calculate_test_scores(y_true: np.ndarray, y_predict: np.ndarray,
                          class_names: [str]) -> {str, float}:

    conf_mat = confusion_matrix(y_true, y_predict)
    conf_mat = pd.DataFrame(conf_mat, columns=class_names, index=class_names)
    msg = "\n" + str(conf_mat)

    class_accuracy = np.round(np.diagonal(conf_mat / np.sum(conf_mat, axis=0)), 3)
    class_accuracy = dict(zip(class_names, class_accuracy))

    msg += "\n\nclass          acc\n"
    line_length = 15
    for name, acc in class_accuracy.items():
        white_space = " " * max(line_length - len(str(name)), 0)
        msg += str(name) + white_space + str(acc) + "\n"

    roc_scores = {"accuracy": round(accuracy_score(y_true, y_predict,), 3),
                  "f1_score": round(f1_score(y_true, y_predict, average="macro"), 3),
                  "precision": round(precision_score(y_true, y_predict, average="macro"), 3),
                  "recall": round(recall_score(y_true, y_predict, average="macro"), 3),
                  }

    rows = [f"{name}: {val}" for name, val in roc_scores.items()]
    msg += "\n" + "\n".join(rows) + "\n"
    print(msg)

    return roc_scores


def print_scores(episode: int, score_dict: {str: float}):
    message = f"Epoch {episode} of {N_EPOCHS}"
    for key, val in score_dict.items():
        message += f" | {key}: {val:.3f}"
    seperator = "-" * len(message)
    if episode == 0:
        message = seperator + "\n" + message + "\n" + seperator
    else:
        message = message + "\n" + seperator
    print(message)


def print_histogram(hist: {str, int}):
    header = "\n"
    header += "class          samples(training data)\n"
    header += "-" * len(header) + "\n"
    msg = header

    line_length = 15
    for label, count in hist.items():
        label_str = str(label)
        count_str = str(count)
        white_space = max(line_length - len(label_str), 0)
        msg += label_str + " " * white_space + count_str + "\n"

    print(msg)


def plot_training_history(history_dict: {str, list}, path: str ):
    if os.path.isdir(path):
        default_file_name = "training_history.pdf"
        path = os.path.join(path, default_file_name)

    fig, axs = plt.subplots(2, 1, figsize=(16, 9))
    data = history_dict["avg_training_loss"]
    axs[0].plot(range(len(data)), data, label="training", linestyle='--', marker='o')
    data = history_dict["avg_validation_loss"]
    axs[0].plot(range(len(data)), data, label="validation", linestyle='--', marker='o')
    axs[0].set_title("Loss")
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("score")
    axs[0].grid()
    axs[0].legend()

    data = history_dict["avg_training_accuracy"]
    axs[1].plot(range(len(data)), data, label="training", linestyle='--', marker='o')
    data = history_dict["avg_validation_accuracy"]
    axs[1].plot(range(len(data)), data, label="validation", linestyle='--', marker='o')
    axs[1].set_title("Accuracy")
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("score")
    axs[1].grid()
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=300)


def plot_class_distribution(training_data: DataLoader,
                            validation_data: DataLoader,
                            test_data: DataLoader,
                            path: str):

    if os.path.isdir(path):
        default_file_name = "class_distribution.pdf"
        path = os.path.join(path, default_file_name)

    fig, axs = plt.subplots(3, 1, figsize=(16, 9))

    hist = count_samples_per_classes(training_data)
    class_names = get_classes_from_dataloader(training_data)
    hist = {class_names[label_id]: count for label_id, count in hist.items()}
    axs[0].bar(hist.keys(), hist.values())
    axs[0].set_ylabel("count")
    axs[0].set_xlabel("class")
    axs[0].set_title("Training Data Histogram")
    training_data_histogram = hist

    hist = count_samples_per_classes(validation_data)
    class_names = get_classes_from_dataloader(validation_data)
    hist = {class_names[label_id]: count for label_id, count in hist.items()}
    axs[1].bar(hist.keys(), hist.values())
    axs[1].set_ylabel("count")
    axs[1].set_xlabel("class")
    axs[1].set_title("Validation Data Histogram")

    hist = count_samples_per_classes(test_data)
    class_names = get_classes_from_dataloader(test_data)
    hist = {class_names[label_id]: count for label_id, count in hist.items()}
    axs[2].bar(hist.keys(), hist.values())
    axs[2].set_ylabel("count")
    axs[2].set_xlabel("class")
    axs[2].set_title("Test Data Histogram")

    plt.tight_layout()
    plt.savefig(path, dpi=300)

    print_histogram(training_data_histogram)


def get_classes_from_dataloader(dataloader: DataLoader):
    if isinstance(dataloader.dataset, torch.utils.data.dataset.Subset):
        return dataloader.dataset.dataset.classes
    else:
        return dataloader.dataset.classes


def count_samples_per_classes(data: DataLoader) -> {int, int}:
    class_histogram = {}
    for _, labels in data:
        labels, counts = torch.unique(labels, return_counts=True)
        for label, count in zip(labels.numpy(), counts.numpy()):
            if label in class_histogram:
                class_histogram[label] += count
            else:
                class_histogram[label] = count

    return class_histogram


def save_model(model: nn.Module, path: str):
    if os.path.isdir(path):
        default_file_name = "vgg16_parameter.pt"
        path = os.path.join(path, default_file_name)

    torch.save(model.state_dict(), path)


def save_scores_and_hyperparameter(scores: {str, int}, path: str):
    if os.path.isdir(path):
        default_file_name = "hyperparameter_and_roc_scores.csv"
        path = os.path.join(path, default_file_name)

    hyperparameter = {"number_of_epochs": N_EPOCHS,
                      "batch_size": BATCH_SIZE,
                      "train/val_split": TRAIN_VALIDATION_SPLIT,
                      "weight_decay": WEIGHT_DECAY,
                      "momentum": MOMENTUM,
                      "learning_rate": LEARN_RATE,
                      "lr_scheduler_step_size": LEARN_RATE_STEP_SIZE,
                      "lr_scheduler_gamma": LEARN_RATE_GAMMA}

    combined = {**hyperparameter, **scores}
    csv_data = {"parameter": list(combined.keys()),
                "values": list(combined.values())}

    pd.DataFrame(csv_data).to_csv(path, index=False)


def main():
    folder_path = make_folder()
    print(f"Created checkpoint folder at \"{folder_path}\"")

    print("Trying to load the CIFAR-10 dataset")
    training_data_loader, validation_data_loader, test_data_loader = load_CIFAR10(batch_size=BATCH_SIZE)

    saving_class_distribution_plot_here = os.path.join(folder_path, "class_distribution.pdf")
    print(f"Saving class-distribution-plot to \"{saving_class_distribution_plot_here}\"")
    plot_class_distribution(training_data=training_data_loader,
                            validation_data=validation_data_loader,
                            test_data=test_data_loader,
                            path=saving_class_distribution_plot_here)

    print("Preparing vgg16")
    classifier = prepare_model()
    optimizer = optim.SGD(classifier.parameters(), lr=LEARN_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=LEARN_RATE_STEP_SIZE, gamma=LEARN_RATE_GAMMA)

    print("Starting model training")
    classifier, history = train_model(model=classifier,
                                      optimizer=optimizer,
                                      scheduler=scheduler,
                                      training_data=training_data_loader,
                                      validation_data=validation_data_loader)

    print("Evaluating model on the test-data-set")
    roc_scores = test_model(classifier, test_data_loader)
    save_meta_data_here = os.path.join(folder_path, "hyperparameter_and_roc_scores.csv")
    print("Saving ROC-scores and Hyperparameter to", save_meta_data_here)
    save_scores_and_hyperparameter(scores=roc_scores, path=save_meta_data_here)

    saving_training_history_plot_here = os.path.join(folder_path, "training_history.pdf")
    print(f"Saving training-history-plot to \"{saving_training_history_plot_here}\"")
    plot_training_history(history, path=saving_training_history_plot_here)
    save_model_params_here = os.path.join(folder_path, "model_params.pt")
    print(f"Saving model parameter to \"{save_model_params_here}\"")
    save_model(classifier, path=save_model_params_here)
    print("Done")


def main_with_args(batch_size: int, lr: float, lr_step: int, lr_gamma: float,
                   momentum: float, w_decay: float) -> {str, float}:

    print("Trying to load the CIFAR-10 dataset")
    training_data_loader, validation_data_loader, test_data_loader = load_CIFAR10(batch_size=batch_size)

    print("Preparing vgg16")
    classifier = prepare_model()
    optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    print("Starting model training")
    classifier, history = train_model(model=classifier,
                                      optimizer=optimizer,
                                      scheduler=scheduler,
                                      training_data=training_data_loader,
                                      validation_data=validation_data_loader)

    print("Evaluating model on the test-data-set")
    roc_scores = test_model(classifier, test_data_loader)
    print("Done")

    params = {"batch_size": batch_size, "learning_rat": lr, "learning_rat_step": lr_step,
              "learning_rat_gamma": lr_gamma, "momentum": momentum, "weight_decay": w_decay}

    params.update(roc_scores)
    return params


def grid_search():
    print("starting the Hyperparameter-Grid-Search")

    collect_results = []
    for batch_size in (32, 128, 256):
        for momentum in (0.85, 0.9, 0.95):
            for lr in (0.0005, 0.001, 0.002):
                for lr_step in (1, 2, 3):
                    for lr_gamma in (0.25, 0.5, 0.75, 1.0):
                        params_and_score = main_with_args(batch_size=batch_size,
                                                          lr=lr,  lr_step=lr_step,
                                                          lr_gamma=lr_gamma, momentum=momentum,
                                                          w_decay=WEIGHT_DECAY)
                        collect_results.append(params_and_score)

        save_gird_search_results_here = os.path.join(os.getcwd(), "grid_search_results.csv")
        print(f"Writing result to \"{save_gird_search_results_here}\"")
        pd.DataFrame(collect_results).to_csv(save_gird_search_results_here)
        print("Done")


if __name__ == '__main__':
    main()
