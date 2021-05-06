import torch
from model_network import ModelNetwork


def predict(inputs):
    window_size = 21
    threshold = 0.5
    model_path = './checkpoints/best_model.pth'
    model = ModelNetwork(window_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # inputs : [batch_size, 1, h, w], where 0-free, 1-obstacle
    outputs = (model(inputs) > threshold).float()
    return outputs


if __name__ == '__main__':
    inputs = None
    outputs = predict(inputs)
