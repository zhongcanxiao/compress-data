import numpy as np

def line_nd(start, end, endpoint=False):
    """
    Generate n-dimensional coordinates for a line between two points using NumPy.

    Parameters:
        start (tuple or list): Starting coordinates of the line.
        end (tuple or list): Ending coordinates of the line.
        endpoint (bool): Whether to include the endpoint. Default is False.
        
    Returns:
        np.ndarray: An array of coordinates for the line.
    """
    start = np.asarray(start)
    end = np.asarray(end)
    num_points = np.linalg.norm(end - start, ord=np.inf) + 1
    if endpoint:
        num_points = np.ceil(num_points)
    else:
        num_points = np.floor(num_points)

    num_points = int(num_points)
    coords = [np.linspace(s, e, num_points, endpoint=endpoint) for s, e in zip(start, end)]
    return np.round(np.stack(coords)).astype(int)
