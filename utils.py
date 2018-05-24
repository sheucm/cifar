import numpy as np
import matplotlib.pyplot as plt

def display_cifar(images, size):
    n = len(images)
    plt.figure()
    plt.gca().set_axis_off()
    im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)])
                    for i in range(size)])
    plt.imshow(im)
    plt.show()


def one_hot(vec, vals):
    n = len(vec)
    out = np.zeros((n, vals))
    for r in range(n):
        out[r, vec[r]] = 1
    return out