"""
@author:     许峻玮
@date:       08/23/2023
"""
import torch
from progbar import Progbar


class Tester:
    """
    Description:
        Tests the model.
    """
    def __init__(self, model, criterion, device, test_loader):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.test_loader = test_loader

    def test(self):

        self.model.eval()
        test_loss = 0
        test_correct = 0

        progbar = Progbar(len(self.test_loader.dataset))

        with torch.no_grad():
            for data, target in self.test_loader:

                data = data.to(self.device)
                target = target.to(self.device)
                output = self.model(data)

                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()

                progbar.add(output.size(0), values=[('acc', test_correct / len(self.test_loader.dataset))])

        test_loss /= len(self.test_loader.dataset)
        test_accuracy = 100. * test_correct / len(self.test_loader.dataset)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
