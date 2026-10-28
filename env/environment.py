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
        

        # observation is dict 
        # {local_state: [u1, v1, v1_u, v1_v,   u2, v2, v2_u, v2_v],
        #  global_state: [x1, y1, z1, v1_x, v1_y, v1_z,   x2, y2, z2, v2_x, v2_y, v2_z]}
        # where 1 - glider, 2 - target
        self.observation_space = gym.spaces.Dict(
            {
                "local_state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
                "global_state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),
                "ego_features": gym.spaces.Box(low=-np.inf, high=np.inf, shape= (6,), dtype=np.float32),
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
        self.r_u = sp.lambdify(cords, r_u)
        self.r_v = sp.lambdify(cords, r_v)

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
    
    def _change_basis_in_tangent_space(self, state, vector, reverse = False):
        """
        Action acceleration vector lies in orthonormal basis (so that equal actions would imply equal acceleration),
        But we need to compute this vector in the tangent space coordinate basis
        Tangent space basis : (f1, f2)
        Orthonormal basis : (e1,e2) = (f1/sqrt(<f1,f1>) , (f2 - alpha*f1)/sqrt(<e2,e2>))
        alpha = <f1,f2>/<f1,f1> 
        So we have a*e1 + b*e2 and we need coordinate components x*f1 + y*f2
        if reverse flag is True, we should compute inverse change of basis (from x*f1 + y*f2 to a*e1 + b*e2)
        """
        u,v = state[0],state[1]
        g = self.g(u,v)
        alpha = g[0,1]/g[0,0]
        e2_norm = np.sqrt(g[1,1] -2*alpha*g[0,1] + alpha*alpha*g[0,0])
        if(reverse == False):
            a,b = vector[0], vector[1]
            y = b/e2_norm
            x = a/np.sqrt(g[0,0]) - b*alpha/e2_norm
            return np.array([x,y])
        else:
            x,y = vector[0], vector[1]
            a = (x+alpha*y)*np.sqrt(g[0,0])
            b = y*e2_norm
            return np.array([a,b])
            
    def step(self, action):
        """
        Execute one simulation step:
        1. Convert action into coordinate acceleration.
        2. Integrate glider and target states using RK4.
        3. Compute current distance, reward, and episode completion flags.

        Returns: (observation, reward, terminated, truncated, info)
        """
        action = self._change_basis_in_tangent_space(self.glider_state, action)
        action *= max(1, 1/np.linalg.norm(action))
        action *= self.maximum_thrust
        self.glider_state = self._rk4_step(self.glider_state, action, 0.01)
        self.target_state = self._rk4_step(self.target_state, np.array([0,0]), 0.01)
        self.time += 0.01
        
        global_state = self.cur_obs["global_state"]
        obs = self._get_observation()
        reward = self._get_reward()
        self._get_termination(global_state[:3], global_state[6:9])
        terminated = self.is_terminated  
        truncated = self.is_truncated   
        info = {
            "reason": self.reason,
            "distance": self.previous_distance
            }           
        return obs, reward, terminated, truncated, info
    
    def _global_velocity_vector (self, state):
        """
        Vector V = (V_x, V_y, V_z) - velocity vector (v_u, v_v) in R^3 space
        dr/dt = dr/du * du/dt + dr/dv * dv/dt
        """
        u,v = state[0], state[1]
        dudt, dvdt = state[2], state[3]
        r_u_vec = np.array(self.r_u(u, v), dtype=np.float32).reshape(-1)
        r_v_vec = np.array(self.r_v(u, v), dtype=np.float32).reshape(-1)

        V = r_u_vec * dudt + r_v_vec * dvdt
        return V




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


    def _get_termination(self, prev_g, prev_t):
        obs = self.cur_obs
        global_state = obs["global_state"]
        g = global_state[:3]
        t = global_state[6:9]
        p0, p1 = prev_g - prev_t, g - t
        seg = p1 - p0
        s = np.clip(-(p0 @ seg) / max(seg @ seg, 1e-12), 0.0, 1.0)
        if np.linalg.norm(p0 + s * seg) < self.complete_region:
            self.is_terminated, self.reason = True, "completed"
        elif max(np.max(np.abs(g)), np.max(np.abs(t))) > self.cord_limit:
            self.is_terminated, self.reason = True, "too far"
        elif self.time >= self.timeout:
            self.is_truncated, self.reason = True, "timeout"
    
    def _get_observation(self):
        """
        Transforms velocity vectors from tangent space coordinate basis (du/dt, dv/dt)
        to orthonormal basis (e1, e2) for policy model observation.
        """
        glider = self.glider_state
        target = self.target_state
        
        V_g = self._global_velocity_vector(glider)
        V_t = self._global_velocity_vector(target)

        # Convert velocity components from coordinate basis to orthonormal basis
        g_v = self._change_basis_in_tangent_space(glider, glider[2:4], True)
        t_v = self._change_basis_in_tangent_space(target, target[2:4], True)

        glider = np.array([glider[0],glider[1],g_v[0],g_v[1]])
        target = np.array([target[0],target[1],t_v[0],t_v[1]])
        
        g_pos = np.array(self.r(glider[0],glider[1])).reshape(-1)
        t_pos = np.array(self.r(target[0],target[1])).reshape(-1)

        
        
        obs = {
            "local_state": np.concatenate([glider, target]).astype(np.float32),
            "global_state": np.concatenate([g_pos,V_g,t_pos,V_t]).astype(np.float32)
        }
        self.cur_obs = obs
        ego_features = self._ego_features()
        obs["ego_features"] = ego_features
        self.cut_obs = obs
        return obs
    
    def _get_reward(self):
        """
        Reward function: reward model for getting closer to the target, punish for moving further away and for time passed.
        """
        distance = self._calculate_distance(self.N)
        delta = self.distance_reward_scale*(self.previous_distance - distance)
        self.previous_distance = distance
        if(self.reason == "completed"):
            delta += 20
        if(self.reason == "too far"):
            delta -= 20
        if(self.reason == "timeout"):
            delta -= 10
        return delta - 0.01

    
    def reset(self, seed = None, options = None):
        """
        Reset environment to initial state.
        Randomly generates initial positions and velocities for glider and target.
        default options are:
        options = {
            'timeout': 10,
            'cord_limit': 5.0,
            'maximum_thrust': 10,
            'complete_region': 1e-2,
            'distance_reward_scale': 2,
            'mu_gl': 0,
            'sigma_gl': cord_limit/2,
            'mu_tg': 0,
            'sigma_tg': self.cord_limit/2,
            'mu_gl_vel': 0,
            'sigma_gl_vel': 0.3,
            'mu_tg_vel': 0,
            'sigma_tg_vel': 0.3,
            'N': 10
        }

        Returns: (observation, info)
        """
        self.is_terminated = True
        self.is_truncated = True
        while(self.is_terminated or self.is_truncated):
            self.is_terminated = False
            self.is_truncated = False
            self.time = 0
            self.reason = "None"
            rng = np.random.default_rng(seed)
            self.timeout = options.get('timeout', 10) if options else 10
            self.cord_limit = options.get('cord_limit', 5.0) if options else 5.0
            self.maximum_thrust = options.get('maximum_thrust', 10) if options else 10
            self.complete_region = options.get('complete_region', 1e-2) if options else 1e-2
            self.distance_reward_scale = options.get('distance_reward_scale', 2) if options else 2
            mu_gl = options.get('mu_gl', 0) if options else 0
            sigma_gl = options.get('sigma_gl', self.cord_limit/2) if options else self.cord_limit/2
            mu_tg = options.get('mu_tg', 0) if options else 0
            sigma_tg = options.get('sigma_tg', self.cord_limit/2) if options else self.cord_limit/2
            mu_gl_vel = options.get('mu_gl_vel', 0) if options else 0
            sigma_gl_vel = options.get('sigma_gl_vel', 0.3) if options else 0.3
            mu_tg_vel = options.get('mu_tg_vel', 0) if options else 0
            sigma_tg_vel = options.get('sigma_tg_vel', 0.3) if options else 0.3

            N = options.get('N', 10) if options else 10
            self.N = N
            self.glider_state = np.concat((rng.normal(mu_gl,sigma_gl, 2),rng.normal(mu_gl_vel,sigma_gl_vel,2)))
            self.target_state = np.concat((rng.normal(mu_tg,sigma_tg, 2),rng.normal(mu_tg_vel,sigma_tg_vel,2)))
            self.previous_distance = self._calculate_distance(N)

            self.rng = rng
            obs = self._get_observation()
            global_state = obs["global_state"]
            self._get_termination(global_state[:3], global_state[6:9])
        

         
        info = {
            "distance": self.previous_distance,
        }     
        return obs, info

    def _ego_features(self):
        g, t = self.glider_state, self.target_state
        global_state = self.cur_obs['global_state']
        gp, tp = global_state[0:3], global_state[6:9]
        ru = np.asarray(self.r_u(g[0], g[1]), float).reshape(-1)
        rv = np.asarray(self.r_v(g[0], g[1]), float).reshape(-1)
        e1 = ru / np.linalg.norm(ru)
        e2 = rv - (rv @ e1) * e1; e2 /= np.linalg.norm(e2)
        
        d = tp - gp
        dv = self._global_velocity_vector(t) - self._global_velocity_vector(g)
        return np.array([d @ e1, d @ e2, np.linalg.norm(d),
                        dv @ e1, dv @ e2,self.timeout - self.time],
                        dtype=np.float32)    


