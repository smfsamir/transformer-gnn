import pdb
import torch
import numpy as np

def dynamic_slice(x, starts, sizes):
    """TODO: fill this in; what is it doing?

    Args:
        x (torch.tensor): B x S x H x D.
        starts (Tuple[int]): some 4d index... not sure what
        sizes (Tuple[int]): Tuple containing a shape. Previously observed (B, S + chunk size, H, D). 

    Returns:
        _type_: Hypothesis: slices into the sequence dimension. 
    """
    starts = [np.clip(starts[i], 0, x.shape[i] - sizes[i]) for i in range(len(starts))] # TODO: check hypothesis that this clip is useful for accessing the very last key in the range. 
    for i, (start, size) in enumerate(zip(starts, sizes)): # looping over the dimensions to get the appropriate slice.
        x = torch.index_select(x, i, torch.tensor(range(start, start + size), device=x.device))
    return x


def map_pt(f, xs):
    """Pytorch version of map.

    Args:
        f (_type_): _description_
        xs (_type_): _description_

    Returns:
        _type_: _description_
    """
    t = [f(x) for x in xs]
    return tuple(map(torch.stack, zip(*t)))

def scan(f, init, xs, length=None):
    """Goes through the sequence a chunk at a time. Takes increments as prescribed by the chunk size

    Args:
        f (_type_): _description_
        init (_type_): ID (index) to start from
        xs (Optional[]):Appears to always be None??? 
        length (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x) # carry is the new index
        ys.append(y)
    return carry, torch.stack(ys)