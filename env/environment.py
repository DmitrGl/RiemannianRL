import gymnasium as gym
from typing import Sequence
import numpy as np
import numpy.typing as npt
import sympy as sp

class GeodesicGliderEnv(gym.Env):
    """
    Environment for glider on Riemann manifold embedded in R^3 Euclid space.
    Agent should learn how to operate glider for it to reach nonstatic destination point with shortest time posssible.
    Agent has acces to information about destination and glider: position and velocity vector.
    Agent can act by choosing glider's thrust, defined with acceleration vector in local orthonormal basis.
    Episode ends with succes, or timeout.
    r_manifold - parametrization r(u,v) (for now only will work with r(u,v) = (u,v,f(u,v)) kind of manifolds)
    """
    def __init__(self, r_manifold: sp.Matrix, cords: Sequence[sp.Symbol]):
        """
        Initialize environment spaces, state vectors, and precompute manifold kinematics.

        r_manifold: SymPy Matrix representing vector r(u,v) = (u, v, f(u,v))
        cords: Sequence of SymPy symbols [u, v]
        """
        super().__init__()

        # positions on manifold and velocities, will be set randomly in reset()
        self.glider_state = np.array([0, 0, 0, 0], dtype= np.float32)
        self.target_state = np.array([0, 0, 0, 0], dtype= np.float32)
        self.previous_distance = 0

        # observation is dict {glider: [u, v, v_u, v_v], target: [u, v, v_u, v_v]}
        self.observation_space = gym.spaces.Dict(
            {
                "glider": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
                "target": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            }
        )
        # action is vector [a_u, a_v]
        self.action_space = gym.spaces.Box(low = np.array([-1, -1]), high= np.array([1, 1]), dtype= np.float32)
        
        self._compute_manifold_kinematics(r_manifold, cords)
    
    def _compute_manifold_kinematics(self, r_manifold: sp.Matrix, cords : Sequence[sp.Symbol]):
        """
        Compute Christoffel symbols and coefficients of first fundamental form and its inverse.
        Compiles analytical SymPy expressions into fast numerical NumPy functions.
        """
        
        u = cords[0]
        v = cords[1]
        r_u = r_manifold.diff(u)
        r_v = r_manifold.diff(v)
        g = sp.Matrix([
            [r_u.dot(r_u), r_u.dot(r_v)],
            [r_v.dot(r_u), r_v.dot(r_v)]   
        ])
        g = sp.simplify(g)
        g_inv = g.inv()
        
        Gamma = [[[0 for _ in range (2)]for _ in range(2)]for _ in range(2)]
        for k in range(2):
            for i in range(2):
                for j in range(2):
                    val = 0
                    for l in range(2):
                        term = (sp.diff(g[j,l], cords[i])+
                                sp.diff(g[i,l],cords[j])-
                                sp.diff(g[i,j],cords[l]))
                        val += sp.Rational(1,2)*g_inv[k,l]*term
                    Gamma[k][i][j] = sp.simplify(val)
        
        self.g = sp.lambdify(cords,g)
        self.g_inv = sp.lambdify(cords,g_inv)
        self.christoffel_funcs = sp.lambdify(cords, Gamma)
        self.cords = cords
        self.r = sp.lambdify(cords, r_manifold)

    def _compute_derivatives(self, state, action):
        """
        Compute state derivatives dS/dt according to geodesic equations with control thrust:
        dS/dt = [du/dt, dv/dt, a_u - Gamma[0][i][j]*v_i*v_j, a_v - Gamma[1][i][j]*v_i*v_j]
        """
        res = np.array(state)
        dudt = state[2]
        dvdt = state[3]
        Gamma = self.christoffel_funcs(state[0],state[1])
        res[0] = dudt
        res[1] = dvdt
        res[2] = action[0] - (Gamma[0][0][0]*dudt*dudt + 2*Gamma[0][0][1]*dudt*dvdt + Gamma[0][1][1]*dvdt*dvdt)
        res[3] = action[1] - (Gamma[1][0][0]*dudt*dudt + 2*Gamma[1][0][1]*dudt*dvdt + Gamma[1][1][1]*dvdt*dvdt)
        return res

    def _rk4_step (self, state, action, dt):
        """
        Numerical solution of the following equation using 4th-order Runge-Kutta (RK4) integrator:
        state = S = [u, v, du/dt, dv/dt] ==> 
        dS/dt = [du/dt, dv/dt, a_u - Gamma[u][i][j] * dx_i/dt * dx_j/dt, a_v - Gamma[v][i][j] * dx_i/dt * dx_j/dt]
        """
        k1 = self._compute_derivatives(state,action)
        k2 = self._compute_derivatives(state + k1*dt/2, action)
        k3 = self._compute_derivatives(state + k2*dt/2, action)
        k4 = self._compute_derivatives(state + k3*dt, action)
        S_next = state + (k1+2*k2+2*k3+k4)*dt/6
        return S_next
    
    def _action_to_coordinate_acceleration(self, state, action):
        """
        Action acceleration vector lies in orthonormal basis (so that equal actions would imply equal acceleration),
        But we need to compute this vector in the tangent space coordinate basis
        Tangent space basis : (f1, f2)
        Orthonormal basis : (e1,e2) = (f1/sqrt(<f1,f1>) , (f2 - alpha*f1)/sqrt(<e2,e2>))
        alpha = <f1,f2>/<f1,f1> 
        So we have a*e1 + b*e2 and we need coordinate components x*f1 + y*f2
        """
        u,v = state[0],state[1]
        g = self.g(u,v)
        alpha = g[0,1]/g[0,0]
        e2_norm = np.sqrt(g[1,1] -2*alpha*g[0,1] + alpha*alpha*g[0,0])
        a,b = action[0], action[1]
        y = b/e2_norm
        x = a/np.sqrt(g[0,0]) - b*alpha/e2_norm
        return np.array([x,y])

    
    def step(self, action):
        """
        Execute one simulation step:
        1. Convert action into coordinate acceleration.
        2. Integrate glider and target states using RK4.
        3. Compute current distance, reward, and episode completion flags.

        Returns: (observation, reward, terminated, truncated, info)
        """
        action = self._action_to_coordinate_acceleration(self.glider_state, action)
        self.glider_state = self._rk4_step(self.glider_state, action, 0.01)
        self.target_state = self._rk4_step(self.target_state, np.array([0,0]), 0.01)

        obs = {
            "glider": self.glider_state, 
            "target": self.target_state
        }
        reward = self._get_reward()
        terminated = False  
        truncated = False   
        info = {}           
        return obs, reward, terminated, truncated, info
    

    def _calculate_distance(self, N):
        """
        To calculate estimated distance between glider and target we divide the line segment in R^3 into N points,
        Map each point onto the manifold via r(u,v), and sum Euclidean distances between consecutive 3D points
        """
        ug,vg = self.glider_state[0], self.glider_state[1]
        ut,vt = self.target_state[0], self.target_state[1]
        g_pos = self.r(ug,vg)
        t_pos = self.r(ut,vt)
        v = (t_pos - g_pos)/(N-1)
        points = np.zeros((N,3))
        dist = 0
        for i in range(N):
            cur = g_pos + i*v
            points[i] = cur.reshape(-1)
            cur = self.r(points[i][0], points[i][1])
            points[i] = cur.reshape(-1)
            if i > 0:
                dist += np.linalg.norm(points[i]-points[i-1])
        return dist
        


    def _get_reward(self):
        """
        Reward function: reward model for getting closer to the target, punish for moving further away and for time passed.
        """
        distance = self._calculate_distance(self.N)
        delta = self.previous_distance - distance
        self.previous_distance = distance
        return delta - 0.01

    
    def reset(self, seed = 42, options = (0,1,10)):
        """
        Reset environment to initial state.
        Randomly generates initial positions and velocities for glider and target.

        Returns: (observation, info)
        """
        rng = np.random.default_rng(seed)
        mu, sigma2, N = options[0:3]
        self.N = N
        self.glider_state = rng.normal(mu,sigma2,4)
        self.target_state = rng.normal(mu,sigma2,4)
        self.previous_distance = self._calculate_distance(N)
        
        self.rng = rng
        obs = {
            "glider": self.glider_state, 
            "target": self.target_state
        }
        info = {}     
        return obs, info

    


