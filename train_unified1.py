import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import sys


# ==================== ОПРЕДЕЛЕНИЕ МОДЕЛИ ====================
class BasicBlock(nn.Module):
    """Базовый блок ResNet с остаточными соединениями"""
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # Первая свертка блока
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # Вторая свертка блока
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Короткое соединение (skip connection)
        self.shortcut = nn.Sequential()
        # Если размерность меняется, добавляем проекцию
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        # Прямой проход через основной путь
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Добавляем короткое соединение
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    """Основная архитектура ResNet"""
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64  # Начальное количество каналов

        # Первый сверточный слой
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Четыре слоя ResNet с разным количеством блоков
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # Уменьшение размера в 2 раза
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # Уменьшение размера в 2 раза
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # Уменьшение размера в 2 раза
        
        # Финальный полносвязный слой для классификации
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """Создает слой из нескольких блоков"""
        strides = [stride] + [1] * (num_blocks - 1)  # Первый блок может уменьшать размер
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes  # Обновляем количество входных каналов для следующего блока
        return nn.Sequential(*layers)

    def forward(self, x):
        # Прямой проход через всю сеть
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # Усредняющее пулинг-слоение
        out = torch.nn.functional.avg_pool2d(out, 4)
        # Вытягиваем в вектор для полносвязного слоя
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    """Создает ResNet-18 архитектуру"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================
class MetricsTracker:
    """Класс для отслеживания метрик обучения"""
    def __init__(self):
        self.train_losses = []  # Потери на обучении
        self.train_accs = []    # Точность на обучении
        self.val_losses = []    # Потери на валидации
        self.val_accs = []      # Точность на валидации

    def update(self, train_loss, train_acc, val_loss, val_acc):
        """Обновляет метрики новой эпохой"""
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)

    def plot_metrics(self):
        """Строит графики потерь и точности"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # График потерь
        ax1.plot(self.train_losses, label='Потери на обучении')
        ax1.plot(self.val_losses, label='Потери на валидации')
        ax1.set_title('Потери по эпохам')
        ax1.set_xlabel('Эпоха')
        ax1.set_ylabel('Потери')
        ax1.legend()
        ax1.grid(True)

        # График точности
        ax2.plot(self.train_accs, label='Точность на обучении')
        ax2.plot(self.val_accs, label='Точность на валидации')
        ax2.set_title('Точность по эпохам')
        ax2.set_xlabel('Эпоха')
        ax2.set_ylabel('Точность (%)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()  # Закрываем график чтобы избежать проблем с отображением


def calculate_metrics(model, dataloader, device, class_names):
    """Вычисляет детальные метрики на датасете"""
    model.eval()  # Переводим модель в режим оценки
    all_preds = []   # Все предсказания
    all_targets = [] # Все истинные метки

    with torch.no_grad():  # Отключаем вычисление градиентов для экономии памяти
        for batch in tqdm(dataloader, desc='Вычисление метрик'):
            images, targets = batch
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)  # Берем класс с максимальной вероятностью

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Отчет по классификации
    print("\n" + "=" * 50)
    print("ОТЧЕТ ПО КЛАССИФИКАЦИИ")
    print("=" * 50)
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4))

    # Матрица ошибок
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Матрица ошибок')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Добавляем текстовые аннотации
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    return all_preds, all_targets


def save_model(model, path='best_model.pth'):
    """Сохраняет веса модели"""
    torch.save({
        'model_state_dict': model.state_dict(),
    }, path)
    print(f"Модель сохранена в {path}")


# ==================== КОД ОБУЧЕНИЯ ====================
# Названия классов CIFAR-10
CLASS_NAMES = ['самолет', 'автомобиль', 'птица', 'кошка', 'олень',
               'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик']


def get_dataloaders(batch_size=128, val_split=0.1):
    """Создает загрузчики данных для обучения, валидации и тестирования"""

    # Аугментация данных для обучения
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Случайное обрезание
        transforms.RandomHorizontalFlip(),     # Случайное отражение по горизонтали
        transforms.ToTensor(),                 # Преобразование в тензор
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Нормализация
    ])

    # Трансформации для валидации и тестирования (без аугментации)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Загрузка датасета CIFAR-10
    print("Загрузка датасета CIFAR-10...")
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )

    # Разделяем обучающую выборку на обучение и валидацию
    val_size = int(val_split * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Применяем тестовые трансформации к валидационной выборке
    val_dataset.dataset.transform = test_transform

    # Создаем загрузчики данных
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Обучающих примеров: {len(train_dataset)}")
    print(f"Валидационных примеров: {len(val_dataset)}")
    print(f"Тестовых примеров: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Обучение на одной эпохе"""
    model.train()  # Режим обучения
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Обучение')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        # Обнуляем градиенты
        optimizer.zero_grad()
        
        # Прямой проход
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Обратный проход и оптимизация
        loss.backward()
        optimizer.step()

        # Считаем статистику
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Обновляем прогресс-бар
        pbar.set_postfix({
            'Потери': f'{running_loss / (batch_idx + 1):.3f}',
            'Точность': f'{100. * correct / total:.2f}%'
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    """Валидация на одной эпохе"""
    model.eval()  # Режим оценки
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Отключаем вычисление градиентов
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def main():
    """Основная функция обучения"""
    parser = argparse.ArgumentParser(description='Обучение на CIFAR-10 с ResNet-18')
    parser.add_argument('--batch_size', type=int, default=128, help='Размер батча')
    parser.add_argument('--epochs', type=int, default=50, help='Количество эпох')
    parser.add_argument('--lr', type=float, default=0.001, help='Скорость обучения')
    parser.add_argument('--val_split', type=float, default=0.1, help='Доля валидации')
    args = parser.parse_args()

    # Конфигурация устройства (GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")

    # Создаем директории
    os.makedirs('./data', exist_ok=True)

    # Получаем загрузчики данных
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, val_split=args.val_split
    )

    # Инициализируем модель
    model = ResNet18(num_classes=10).to(device)
    print(f"Создана модель с {sum(p.numel() for p in model.parameters()):,} параметрами")

    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # Уменьшение LR каждые 20 эпох

    # Трекер метрик
    metrics = MetricsTracker()

    # Цикл обучения
    best_val_acc = 0.0  # Лучшая точность на валидации

    print("Начало обучения...")
    for epoch in range(args.epochs):
        print(f'\nЭпоха {epoch + 1}/{args.epochs}')
        print('-' * 50)

        # Обучение
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Валидация
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        # Обновляем метрики
        metrics.update(train_loss, train_acc, val_loss, val_acc)

        # Обновляем планировщик обучения
        scheduler.step()

        print(f'Потери на обучении: {train_loss:.4f}, Точность на обучении: {train_acc:.2f}%')
        print(f'Потери на валидации: {val_loss:.4f}, Точность на валидации: {val_acc:.2f}%')

        # Сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, 'best_model.pth')

    # Строим графики обучения
    print("Построение графиков метрик...")
    metrics.plot_metrics()

    # Загружаем лучшую модель и оцениваем на тестовой выборке
    print("\nОценка на тестовой выборке с лучшей моделью...")
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Вычисляем детальные метрики
    test_preds, test_targets = calculate_metrics(model, test_loader, device, CLASS_NAMES)

    print(f"\nЛучшая точность на валидации: {best_val_acc:.2f}%")
    print("Обучение завершено!")
    print("Сгенерированные файлы:")
    print("- training_metrics.png: Графики потерь и точности во время обучения")
    print("- confusion_matrix.png: Матрица ошибок предсказаний на тесте")
    print("- best_model.pth: Сохраненные веса модели")


if __name__ == '__main__':
    main()