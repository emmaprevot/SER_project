import torch
from ser.transforms import transforms, normalize, flip

def test_trasform_flip():
    """
    Flip is meant to flip an image both horizontally vertically.
    I.e. it flips it along a diagonal axis, turning a 6 into a 9.
    A good example is:
      1 0 1
      0 1 0
      0 0 1
    We expect that to be flipped into:
      1 0 0
      0 1 0
      1 0 1
    So lets test that!
    """
    input = torch.FloatTensor([[[1, 0, 1], [0, 1, 0], [0, 0, 1]]])
    output = torch.FloatTensor([[[1, 0, 0], [0, 1, 0], [1, 0, 1]]])
    assert torch.equal(flip()(input), output)
    
    
def test_trasform_normalize():
    input = torch.FloatTensor([[[1, 1, 1]]])
    output = torch.FloatTensor([[[1, 1, 1]]])
    assert torch.equal(normalize()(input), output)