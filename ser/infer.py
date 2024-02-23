import torch

from ser.utils import generate_ascii_art, pixel_to_char

def infer(run_path, label, dataloader):
    
    images, labels = next(iter(dataloader))
    while labels[0].item() != label:
        images, labels = next(iter(dataloader))

    # load the model
    model = torch.load(run_path / "model.pt")

    # run inference
    model.eval()
    output = model(images)
    pred = output.argmax(dim=1, keepdim=True)[0].item()
    confidence = max(list(torch.exp(output)[0]))
    pixels = images[0][0]
    print(generate_ascii_art(pixels))
    print(f"This is a {pred}")
    print("Inference is finished!")