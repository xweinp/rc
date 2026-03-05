import numpy as np
from tqdm import tqdm

from transformation import add_ones, find_transformation

EPS = 1e-5


def gen_H():
    while True:
        H = np.random.rand(3, 3)
        if abs(np.linalg.det(H)) > EPS:
            return H


def transformation_test():
    points1 = np.random.randint(0, 256, (40, 2))
    H = gen_H()
    points2 = (H @ add_ones(points1).T).T
    points2 = points2[:, :2] / points2[:, 2:3]

    H_recovered = find_transformation(points1, points2)
    try:
        np.testing.assert_allclose(
            H / H[-1, -1], H_recovered / H_recovered[-1, -1], atol=1e-2, rtol=1e-2
        )
    except AssertionError as e:
        print("Test failed!")
        print("Original H:")
        print(H / H[-1, -1])
        print("Recovered H:")
        print(H_recovered / H_recovered[-1, -1])
        return False
    return True


def main():
    print("Testing projective transformation...")
    for _ in tqdm(range(100)):
        if not transformation_test():
            return
    print("All tests passed!")


if __name__ == "__main__":
    main()
