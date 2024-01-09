import torch


class AudioPredictor:
    @staticmethod
    def predict(model, input):
        model.eval()

        input = torch.clone(input)
        input = input.unsqueeze_(0)

        with torch.no_grad():
            pred = model(input)
            pred_value = pred[0].argmax(0)

        return pred_value.item()
