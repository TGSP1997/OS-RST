import numpy as np

# generate gaussian white noise
def white_noise(mean, std_dev, point_count, seed):    
    np.random.seed(seed)
    noise = np.random.normal(mean, std_dev, point_count)
    return noise