import torch

from ser.utils import generate_ascii_art, pixel_to_char, _select_test_image

def infer(model, params, label, dataloader):
    
    print("Runninf inference for the following model:")
    for i in params:
        print(i + ": " + str(params[i]))
        
    images = _select_test_image(dataloader, label)

    # run inference
    model.eval()
    output = model(images)
    pred = output.argmax(dim=1, keepdim=True)[0].item()
    confidence = max(list(torch.exp(output)[0]))

    pixels = images[0][0]
    print(generate_ascii_art(pixels))
    
    print(f"I am {confidence * 100:.2f}% confident that it's a... {pred}\n")
    print("Inference is finished!")


