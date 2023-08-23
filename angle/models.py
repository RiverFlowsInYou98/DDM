import numpy as np

# converge strongly with order k: E[|X_T-Y_N|] <= C * dt^k, this concerns the convergence of the sample path
# converge weakly with order k: |E[g(X_T)]-E[g(Y_N)]| <= C * dt^k, this concerns the convergence of the distribution

class AngleModel(object):
    """
    simulate the two-sided first passage time of angle model
    """
    def __init__(self, mu=1, sigma=1) -> None:
        self.mu = mu
        self.sigma = sigma

    def drift_coeff(self, _X: float, _t: float) -> float:
        return self.mu

    def diffu_coeff(self, _X: float, _t: float) -> float:
        return self.sigma

    def dW(self, dt: float) -> float:
        return np.random.normal(loc=0.0, scale=np.sqrt(dt))

    def EulerMaruyama(self, init_cond, T, Nt=1000):
        # Euler-Maruyama scheme, strong order 0.5, weak order 1.0
        t0, X0 = init_cond
        dt = float(T - t0) / Nt
        t_grid = np.arange(t0, T + dt, dt)
        X_grid = np.zeros(Nt + 1)
        X_grid[0] = X0
        for i in range(1, Nt + 1):
            t = t0 + (i - 1) * dt
            X = X_grid[i - 1]
            X_grid[i] = (
                X + self.drift_coeff(X, t) * dt + self.diffu_coeff(X, t) * self.dW(dt)
            )
        return t_grid, X_grid

    def Milstein(self, init_cond, T, Nt=1000):
        # Milstein scheme, strong order 0.5, weak order 1.0
        t0, X0 = init_cond
        dt = float(T - t0) / Nt
        t_grid = np.arange(t0, T + dt, dt)
        X_grid = np.zeros(Nt + 1)
        X_grid[0] = X0
        for i in range(1, Nt + 1):
            t = t0 + (i - 1) * dt
            X = X_grid[i - 1]
            X_grid[i] = (
                X
                + self.drift_coeff(X, t) * dt
                + self.diffu_coeff(X, t) * self.dW(dt)
                + 0.5 * self.diffu_coeff(X, t) ** 2 * (self.dW(dt) ** 2 - dt)
            )
        return t_grid, X_grid

    def LeimkuhlerMatthews(self, init_cond, T, Nt=1000):
        t0, X0 = init_cond
        dt = float(T - t0) / Nt
        t_grid = np.arange(t0, T + dt, dt)
        X_grid = np.zeros(Nt + 1)
        X_grid[0] = X0
        for i in range(1, Nt + 1):
            t = t0 + (i - 1) * dt
            X = X_grid[i - 1]
            X_temp = (
                X
                + self.drift_coeff(X, t) * dt
                + 0.5 * self.diffu_coeff(X, t) * self.dW(dt)
            )
            X_grid[i] = X_temp + 0.5 * self.diffu_coeff(X, t) * self.dW(dt)
        return t_grid, X_grid

    def simulate_FPTD(self, init_cond, T, a, theta, dt=0.001, num=3000):
        # may use importance sampling to imporve...
        # upper and lower boundaries
        upper_bdy = lambda t: a - theta * t
        lower_bdy = lambda t: -a + theta * t
        # use Milstein scheme
        t0, X0 = init_cond
        upper_time, lower_time, vertical_x = [], [], []
        for n in range(num):
            
            # print(n, end=" ")
            
            X = X0
            for i in range(1, int((T-t0)/dt)+10):
                t = t0 + (i - 1) * dt
                X += (
                    self.drift_coeff(X, t) * dt
                    + self.diffu_coeff(X, t) * self.dW(dt)
                    + 0.5 * self.diffu_coeff(X, t) ** 2 * (self.dW(dt) ** 2 - dt)
                )
                if X >= upper_bdy(t):
                    upper_time.append(t)
                    break
                elif X <= lower_bdy(t):
                    lower_time.append(t)
                    break
                elif t>=T:
                    vertical_x.append(X)
                    break
        return np.array(upper_time), np.array(lower_time), np.array(vertical_x)
    
class Bm_wDrift(object):
    """
    simulate the one-sided first passage time of Brownian motion with drift and scaling
    """

    def __init__(self, mu=1, sigma=1) -> None:
        self.mu = mu
        self.sigma = sigma

    def drift_coeff(self, _X: float, _t: float) -> float:
        return self.mu

    def diffu_coeff(self, _X: float, _t: float) -> float:
        return self.sigma

    def dW(self, dt: float) -> float:
        return np.random.normal(loc=0.0, scale=np.sqrt(dt))

    def simulate_FPTD(self, init_cond, T, a, dt=0.001, num=3000):
        # may use importance sampling to imporve...
        # upper and lower boundaries
        bdy = lambda t: a
        t0, X0 = init_cond
        sign = np.sign(X0 - bdy(0))
        passage_time, vertical_x = [], []
        for n in range(num):
            X = X0
            # use Milstein scheme
            for i in range(1, int((T - t0) / dt) + 10):
                t = t0 + (i - 1) * dt
                X += (
                    self.drift_coeff(X, t) * dt
                    + self.diffu_coeff(X, t) * self.dW(dt)
                    + 0.5 * self.diffu_coeff(X, t) ** 2 * (self.dW(dt) ** 2 - dt)
                )
                if (X - bdy(t)) * sign <= 0:
                    passage_time.append(t)
                    break
                elif t >= T:
                    vertical_x.append(X)
                    break
        return np.array(passage_time), np.array(vertical_x)