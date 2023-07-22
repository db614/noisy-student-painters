
import torch
import torch.optim as optim


def train(model, training_data, validation_data, device, criterion, lr=0.001, momentum=0.9, epochs=2, save=False,
          save_path=None):

    model = model.to(device)

    criterion = criterion.to(device)
    optimizer = optim.SGD(model.parameters(), lr, momentum)

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0
        for i, data in enumerate(training_data, 0):
            # data is a list of [imgs, labels, image_id]
            inputs, labels, _ = data
            inputs = inputs.to(device)
            labels = labels.long().to(device)
            optimizer.zero_grad()

            # output, calculate loss, prop backward, step optimiser forwards
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        validation_loss = 0.0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(validation_data, 0):
                # data is a list of [imgs, labels, image_id]
                inputs, labels, _ = data
                inputs = inputs.to(device)
                labels = labels.long().to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                validation_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {validation_loss / 2000:.3f}')
                    validation_loss = 0.0

    print('Finished Training')

    if save:
        torch.save(model.state_dict(), save_path)
