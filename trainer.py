"""
@author:     许峻玮
@date:       08/23/2023
"""
import torch
from progbar import Progbar


class Trainer:
    """
    Arguments:
        - model: model to train
        - optimizer: optimizer to use
        - criterion: loss function
        - device: device to use
        - train_loader: training data loader
        - test_loader: test data loader
        - epochs: number of epochs to train
    Description:
        Trains the model.
    """

    def __init__(self, model, optimizer, criterion, device, train_loader, test_loader, scheduler=None, epochs=10):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.train_acc_list = []
        self.test_acc_list = []
        self.train_loss_list = []
        self.test_loss_list = []

    def train(self):

        prev_loss = float('inf')

        for epoch in range(self.epochs):
            print(f'[Epoch {epoch + 1}/{self.epochs}]')

            self.model.train()
            train_loss = 0
            train_correct = 0

            progbar = Progbar(len(self.train_loader.dataset))

            for _, (data, target) in enumerate(self.train_loader):
                data = data.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)

                loss = self.criterion(output, target)
                train_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()

                loss.backward()
                self.optimizer.step()

                progbar.add(output.size(0), values=[('loss', loss.item())])
            
            train_loss /= len(self.train_loader.dataset)
            train_accuracy = 100. * train_correct / len(self.train_loader.dataset)
            print(f'Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.2f}%')

            self.model.eval()
            test_loss = 0
            test_correct = 0

            with torch.no_grad():
                for data, target in self.test_loader:
                    data = data.to(self.device)
                    target = target.to(self.device)
                    output = self.model(data)

                    test_loss += self.criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    test_correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(self.test_loader.dataset)
            test_accuracy = 100. * test_correct / len(self.test_loader.dataset)
            print(f'Test Loss: {test_loss:.6f}, Test Accuracy: {test_accuracy:.2f}%')

            if test_loss < prev_loss:
                prev_loss = test_loss
                torch.save(self.model.state_dict(), 'model.pt')
            
            self.train_acc_list.append(train_accuracy)
            self.test_acc_list.append(test_accuracy)
            self.train_loss_list.append(train_loss)
            self.test_loss_list.append(test_loss)
            
            if self.scheduler is not None:
                self.scheduler.step()
