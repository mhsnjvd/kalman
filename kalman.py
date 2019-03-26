import numpy as np
import matplotlib.pylab as plt

def get_measurement_noise(n, sigma):
    """Get the measurement noise as an n-vector
    :param n: a positive integer
    :return: a numpy array with random noise
    """
    v = sigma * np.random.randn(n)
    return v

def get_process_noise(n, sigma):
    """Get the process noise as an n-vector
    :param n: a positive integer
    :return: a numpy array with random noise
    """
    w = sigma * np.random.randn(n)
    return w

def get_process_noise_variance_matrix(n, sigma):
    """Get variance matrix.
    :param n:
    :param sigma:
    :return:
    """
    Q = sigma * sigma * np.eye(n)
    return Q

def get_measurement_noise_variance_matrix(n, sigma):
    """Get variance matrix.
    :param n:
    :param sigma:
    :return:
    """
    R = sigma * sigma * np.eye(n)
    return R



def get_process_matrix(dt=1e-3):
    """Retun the process matrix F
    :return:
    """
    F = np.array([[1.0, dt, dt*dt/2.0], [0, 1.0, dt], [0.0, 0.0, 1]])
    return F

def get_measurement_matrix():
    """Return the measurement matrix H
    :return:
    """
    H = np.zeros((3, 3))
    H[0][0] = 1
    return H

def get_control_matrix():
    """Return the control matrix
    :return:
    """
    # For now we are not using any control input
    return np.zeros((3,3))

def get_prediction(F, P_km1_km1, B, u, x_km1_km1, Q):
    x_k_km1 = F.dot(x_km1_km1) + B.dot(u)
    P_k_km1 = np.matmul(F, np.matmul(P_km1_km1, F.transpose())) + Q
    return x_k_km1, P_k_km1

def update_filter(H, P, x_k_km1, y_k):
    z_k = y_k - H.dot(x_k_km1)
    S_k = np.matmul(H, np.matmul(P, H.transpose())) + get_measurement_noise(len(y_k))
    return z_k, S_k


def apply_kalman_gain(P, H, S, z):
    """We need to computed P * H^T * S^-1 * z
    :param P:
    :param H:
    :param S:
    :param z:
    :return: P * H^T * S^-1 * z
    """
    # Compute y = S^-1 * z:
    y = np.linalg.solve(S, z)
    return P.dot((H.transpose()).dot(y))

def get_kalman_gain(P, H, S):
    """Compute the matrix P * H^T * S^-1
    :param P:
    :param H:
    :param S:
    :return:
    """
    Sinv = np.linalg.solve(S, np.eye(S.shape[0]))
    return np.matmul(P, np.matmul(H, Sinv))

def innovate(x_k_km1, z_k, P, H, S):
    """The innovation step
    :param x_k_km1:
    :param z_k:
    :param P:
    :param H:
    :param S:
    :return:
    """
    x_k_k = x_k_km1 + apply_kalman_gain(P, H, S, z_k)
    return x_k_k

def update_variance(K, H, P):
    I = np.eye(K.shape[0])
    P_new = np.matmul(I - np.matmul(K, H), P)
    return P_new

def main():
    pass


if __name__=="__main__":
    T = 1
    N = 1001
    dt = 1/(N-1)
    t = np.linspace(0, 1, N)
    x0 = 0.0
    v0 = 1.0
    a0 = 0.0
    sigma_process = 1e-2
    sigma_measurement = 1e-1

    # Store the simulated states
    m = 3
    X = np.zeros((m, N))
    # Initial state vector
    x_init = np.array([x0, v0, a0])
    X[:, 0] = x_init

    # Store the simulated measurements
    Z = np.zeros((m, N))
    Zhat = np.zeros((m, N))

    F = get_process_matrix(dt)
    H = get_measurement_matrix()
    Q = get_process_noise_variance_matrix(m, sigma_process)
    R = get_measurement_noise_variance_matrix(m, sigma_measurement)
    B = 0 * F
    u = 0 * x_init

    P = [None] * N
    P[0] = np.zeros(Q.shape)

    S = [None] * N
    S[0] = np.eye(m)


    X_predicted = np.zeros((m, N))
    X_updated   = np.zeros((m, N))

    for i in range(1, N):
        X[:, i] =  F.dot(X[:, i-1]) + get_process_noise(m, sigma_process)
        X_predicted[:, i], P[i] = get_prediction(F, P[i - 1], B, u, X[:, i - 1], Q)
        Z[:, i] =  H.dot(X[:, i]) + get_measurement_noise(m, sigma_measurement)
        Zhat[:, i] =  Z[:, i] - H.dot(X_predicted[:, i])
        S[i] = np.matmul(H, np.matmul(P[i], H.transpose())) + R
        K = get_kalman_gain(P[i], H, S[i])
        X_updated[:, i] = X_predicted[:, i] + K.dot(Zhat[:, i])
        P[i] = update_variance(K, H, P[i])



    X_internal = X[0, :]
    X_observed = Z[0, :]
    X_final    = X_updated[0, :]

    print(X_internal)
    print(X_observed)
    plt.plot(t, X_internal, 'b.-', t, X_observed, 'r.', t, X_final, 'k*')
    plt.ylabel('Position (m)')
    plt.xlabel('time (s)')
    plt.title('Observed and internal position as a function of time')

    plt.grid()
    plt.show()
