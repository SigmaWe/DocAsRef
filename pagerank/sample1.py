import numpy as np


if __name__ == "__main__":
    w_refs = np.array([10, 20, 30])
    dist_mat = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    # wd_pairs = [w_refs[i] * dist_mat[i][j] for i in range(3) for j in range(3)]
    # wd_mat = np.array(wd_pairs).reshape((3, 3))
    wd_mat = (dist_mat.T * w_refs).T
    # v = [np.sum(wd_mat[i]) for i in range(3)]
    v = np.sum(wd_mat, axis=1).tolist()
    print(v)
