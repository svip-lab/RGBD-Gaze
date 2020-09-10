import cvxpy as cp
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt


def scatter_face_landmark(face_landmark, median, median_mask):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('D')
    xs = face_landmark[median_mask, 0]
    ys = face_landmark[median_mask, 1]
    zs = median[median_mask]
    ax.scatter(xs, ys, zs, c="r", marker="x")
    ax.view_init(-109, -64)
    return fig, ax


class DepthSolver(object):
    def __init__(self, solver=cp.OSQP):
        self.solver = solver

    def solve(self, rel_depth_pred, rel_depth_mask, depth_obs, depth_obs_mask, dist_landmark, sigma_dist, lambda_E_D,
              lambda_E_rel, lambda_E_s):
        E_D, E_rel, E_s = 0, 0, 0
        bs, num_landmarks = len(rel_depth_pred), len(depth_obs[0])
        depth_pred = cp.Variable(value=np.ones((bs, num_landmarks)) * 750, shape=(bs, num_landmarks), nonneg=True)
        for i in range(bs):
            D = depth_pred[i]
            D_stack1 = cp.atoms.affine.transpose.transpose(cp.atoms.affine.vstack.vstack([D] * num_landmarks))
            D_stack2 = cp.atoms.affine.vstack.vstack([D] * num_landmarks)
            D_err = (D_stack1 - D_stack2)
            if lambda_E_D > 0:
                E_D += cp.sum_squares(cp.atoms.affine.binary_operators.multiply(
                    depth_obs[i] - depth_pred[i], depth_obs_mask[i]
                ))
            if lambda_E_rel > 0:
                E_rel += cp.sum_squares(cp.atoms.affine.binary_operators.multiply(
                    D_err - rel_depth_pred[i] * dist_landmark[i], rel_depth_mask[i]
                ))
            if lambda_E_s > 0:
                E_s += cp.sum_squares(cp.atoms.affine.binary_operators.multiply(
                    D_err, np.exp(-dist_landmark[i] ** 2 / sigma_dist ** 2)
                ))
        obj = cp.Minimize(E_D * lambda_E_D + E_rel * lambda_E_rel + E_s * lambda_E_s)
        prob = cp.Problem(obj, [D >= 0])
        assert prob.is_dcp() and prob.is_qp()

        prob.solve(solver=self.solver, verbose=False)

        if prob.status != cp.OPTIMAL:
            raise Exception("Solver did not converge!")
        res = np.stack([d.value for d in depth_pred], axis=0)
        return res
