

import warnings
from unittest.mock import patch, MagicMock
import numpy as np
import sympy as sp
from sympy import symbols, Function, Eq, diff, lambdify, sin, cos, exp, pi, sympify

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_banded
import warnings

warnings.filterwarnings('ignore')

class SymbolicNumericPDESolver:
    """
    A hybrid solver that accepts symbolic PDE definitions and solves them numerically.
    Supports 1D and 2D PDEs with Dirichlet and Neumann boundary conditions.
    """
    
    def __init__(self):
        self.pde = None
        self.domain = None
        self.boundary_conditions = []
        self.initial_condition = None
        self.solution = None
        self.mesh = None
        self.time_dependent = False
        
    # def define_pde_notuse(self, pde_expr, dependent_var, independent_vars, domain_bounds):
    #     """
    #     Define the PDE symbolically.
        
    #     Args:
    #         pde_expr: SymPy expression representing the PDE (should equal 0)
    #         dependent_var: The dependent variable (e.g., u)
    #         independent_vars: List of independent variables (e.g., [x, t] or [x, y])
    #         domain_bounds: Dictionary with bounds for each independent variable
    #                       e.g., {'x': (0, 1), 't': (0, 1)} or {'x': (0, 1), 'y': (0, 1)}
    #     """
    #     self.pde = pde_expr
    #     self.dependent_var = dependent_var
    #     self.independent_vars = independent_vars
    #     self.domain_bounds = domain_bounds
        
    #     # Check if time-dependent
    #     self.time_dependent = any(str(var) == 't' for var in independent_vars)
        
    #     print(f"PDE defined: {pde_expr} = 0")
    #     print(f"Dependent variable: {dependent_var}")
    #     print(f"Independent variables: {independent_vars}")
    #     print(f"Domain bounds: {domain_bounds}")
        

    def define_pde(self, pde_expr, dependent_var, independent_vars, domain_bounds):
        """
        Define the PDE symbolically.

        Args:
            pde_expr: SymPy expression representing the PDE (should equal 0)
            dependent_var: The dependent variable (e.g., u)
            independent_vars: List of independent variables (e.g., [x, t] or [x, y])
            domain_bounds: Dictionary with bounds for each independent variable
                        e.g., {'x': (0, 1), 't': (0, 1)} or {'x': (0, 1), 'y': (0, 1)}
        """
        self.pde = pde_expr
        self.dependent_var = dependent_var
        self.independent_vars = independent_vars
        self.domain_bounds = domain_bounds

        # Check for undefined symbols in the PDE
        allowed_symbols = set(independent_vars) | {dependent_var}
        used_symbols = pde_expr.free_symbols
        if not used_symbols.issubset(allowed_symbols):
            raise ValueError(f"PDE contains undefined variables: {used_symbols - allowed_symbols}")

        # Check if time-dependent
        self.time_dependent = any(str(var) == 't' for var in independent_vars)

        print(f"PDE defined: {pde_expr} = 0")
        print(f"Dependent variable: {dependent_var}")
        print(f"Independent variables: {independent_vars}")
        print(f"Domain bounds: {domain_bounds}")

    # def add_boundary_condition_notuse(self, condition_type, location, value_expr, variable=None):
    #     """
    #     Add boundary conditions.
        
    #     Args:
    #         condition_type: 'dirichlet' or 'neumann'
    #         location: Dictionary specifying boundary location, e.g., {'x': 0} or {'x': 1}
    #         value_expr: SymPy expression for the boundary value
    #         variable: For Neumann conditions, specify which variable to differentiate
    #     """
    #     bc = {
    #         'type': condition_type,
    #         'location': location,
    #         'value': value_expr,
    #         'variable': variable
    #     }
    #     self.boundary_conditions.append(bc)
    #     print(f"Added {condition_type} BC at {location}: {value_expr}")
        
    def add_boundary_condition(self, condition_type, location, value_expr, variable=None):
        """
        Add boundary conditions.

        Args:
            condition_type: 'dirichlet' or 'neumann'
            location: Dictionary specifying boundary location, e.g., {'x': 0} or {'x': 1}
            value_expr: SymPy expression for the boundary value
            variable: For Neumann conditions, specify which variable to differentiate
        """
        supported_types = {'dirichlet', 'neumann'}
        if condition_type not in supported_types:
            raise ValueError(f"Unsupported boundary condition: {condition_type}")

        bc = {
            'type': condition_type,
            'location': location,
            'value': value_expr,
            'variable': variable
        }
        self.boundary_conditions.append(bc)
        print(f"Added {condition_type} BC at {location}: {value_expr}")

    def set_initial_condition(self, initial_expr):
        """Set initial condition for time-dependent problems."""
        self.initial_condition = initial_expr
        print(f"Initial condition: {initial_expr}")
        
    # def create_mesh_notuse(self, resolution):
    #     """
    #     Create mesh grid based on domain bounds and resolution.
        
    #     Args:
    #         resolution: Dictionary with number of points for each dimension
    #                    e.g., {'x': 50, 't': 100} or {'x': 50, 'y': 50}
    #     """
    #     self.resolution = resolution
    #     self.mesh = {}
        
    #     for var_name, var in zip([str(v) for v in self.independent_vars], self.independent_vars):
    #         if var_name in self.domain_bounds:
    #             bounds = self.domain_bounds[var_name]
    #             n_points = resolution.get(var_name, 50)
    #             self.mesh[var_name] = np.linspace(bounds[0], bounds[1], n_points)
                
    #     print(f"Mesh created with resolution: {resolution}")

    def create_mesh(self, resolution):
        """
        Create mesh grid based on domain bounds and resolution.

        Args:
            resolution: Dictionary with number of points for each dimension
                        e.g., {'x': 50, 't': 100} or {'x': 50, 'y': 50}
        """
        # Validate that all keys in resolution are valid independent variables
        for var in resolution:
            if var not in self.domain_bounds:
                raise ValueError(f"Resolution contains variable '{var}' which is not in domain_bounds: {list(self.domain_bounds.keys())}")

        self.resolution = resolution
        self.mesh = {}

        for var_name, var in zip([str(v) for v in self.independent_vars], self.independent_vars):
            if var_name in self.domain_bounds:
                bounds = self.domain_bounds[var_name]
                n_points = resolution.get(var_name, 50)
                self.mesh[var_name] = np.linspace(bounds[0], bounds[1], n_points)

        print(f"Mesh created with resolution: {resolution}") 

    def _finite_difference_1d(self):
        """Solve 1D PDE using finite differences."""
        x_vals = self.mesh['x']
        dx = x_vals[1] - x_vals[0]
        n = len(x_vals)
        
        if self.time_dependent:
            return self._solve_time_dependent_1d(dx)
        else:
            return self._solve_steady_state_1d(dx)
            
    def _solve_steady_state_1d_notuse(self, dx):
        """Solve steady-state 1D PDE."""
        x_vals = self.mesh['x']
        n = len(x_vals)
        
        # Create coefficient matrix for finite differences
        # For second derivative: u''(x) ≈ (u[i+1] - 2*u[i] + u[i-1])/dx²
        A = np.zeros((n, n))
        b = np.zeros(n)
        
        # Interior points
        for i in range(1, n-1):
            # Second derivative coefficient
            A[i, i-1] = 1/dx**2
            A[i, i] = -2/dx**2
            A[i, i+1] = 1/dx**2
            
        # Apply boundary conditions
        for bc in self.boundary_conditions:
            if 'x' in bc['location']:
                x_bc = bc['location']['x']
                idx = np.argmin(np.abs(x_vals - x_bc))
                
                if bc['type'] == 'dirichlet':
                    A[idx, :] = 0
                    A[idx, idx] = 1
                    b[idx] = float(bc['value'])
                elif bc['type'] == 'neumann':
                    # du/dx ≈ (u[i+1] - u[i-1])/(2*dx) = value
                    if idx == 0:  # Left boundary
                        A[idx, 0] = -3/(2*dx)
                        A[idx, 1] = 4/(2*dx)
                        A[idx, 2] = -1/(2*dx)
                    elif idx == n-1:  # Right boundary
                        A[idx, n-3] = 1/(2*dx)
                        A[idx, n-2] = -4/(2*dx)
                        A[idx, n-1] = 3/(2*dx)
                    b[idx] = float(bc['value'])
        
        # Solve the linear system
        try:
            solution = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(A, b, rcond=None)[0]
            
        return solution

    # def _solve_steady_state_1d(self, dx):
    #     x_vals = self.mesh['x']
    #     n = len(x_vals)

    #     A = np.zeros((n, n))
    #     b = np.zeros(n)

    #     # Infer source term f(x) from PDE: assume form d²u/dx² + f(x) = 0
    #     u = self.dependent_var
    #     x = self.independent_vars[0]
    #     f_expr = -self.pde.subs(sp.Derivative(u, x, 2), 0).doit()
    #     f_func = lambdify(x, f_expr, 'numpy')
        
    #     # Interior points
    #     for i in range(1, n - 1):
    #         A[i, i - 1] = 1 / dx**2
    #         A[i, i] = -2 / dx**2
    #         A[i, i + 1] = 1 / dx**2
    #         b[i] = f_func(x_vals[i])

    #     # Apply boundary conditions
    #     for bc in self.boundary_conditions:
    #         if 'x' in bc['location']:
    #             x_bc = bc['location']['x']
    #             idx = np.argmin(np.abs(x_vals - x_bc))
    #             if bc['type'] == 'dirichlet':
    #                 A[idx, :] = 0
    #                 A[idx, idx] = 1
    #                 b[idx] = float(bc['value'])
    #             elif bc['type'] == 'neumann':
    #                 if idx == 0:
    #                     A[idx, 0] = -3 / (2 * dx)
    #                     A[idx, 1] = 4 / (2 * dx)
    #                     A[idx, 2] = -1 / (2 * dx)
    #                 elif idx == n - 1:
    #                     A[idx, n - 3] = 1 / (2 * dx)
    #                     A[idx, n - 2] = -4 / (2 * dx)
    #                     A[idx, n - 1] = 3 / (2 * dx)
    #                 b[idx] = float(bc['value'])

    #     try:
    #         solution = np.linalg.solve(A, b)
    #     except np.linalg.LinAlgError:
    #         solution = np.linalg.lstsq(A, b, rcond=None)[0]

    #     return solution

    def _solve_steady_state_1d(self, dx):
        x_vals = self.mesh['x']
        n = len(x_vals)

        A = np.zeros((n, n))
        b = np.zeros(n)

        # Infer source term f(x) from PDE: assume form d²u/dx² + f(x) = 0
        u = self.dependent_var
        x = self.independent_vars[0]

        try:
            f_expr = -self.pde.subs(sp.Derivative(u, x, 2), 0).doit()
            if f_expr.has(sp.Derivative):
                raise ValueError("Unsupported: Source term contains unresolved derivatives.")
            f_func = lambdify(x, f_expr, 'numpy')
        except Exception as e:
            print(f"[Warning] Source term not parsable; defaulting to f(x)=0. Reason: {e}")
            f_func = lambda x_val: 0

        # Interior points
        for i in range(1, n - 1):
            A[i, i - 1] = 1 / dx**2
            A[i, i] = -2 / dx**2
            A[i, i + 1] = 1 / dx**2
            b[i] = f_func(x_vals[i])

        # Apply boundary conditions
        for bc in self.boundary_conditions:
            if 'x' in bc['location']:
                x_bc = bc['location']['x']
                idx = np.argmin(np.abs(x_vals - x_bc))
                if bc['type'] == 'dirichlet':
                    A[idx, :] = 0
                    A[idx, idx] = 1
                    b[idx] = float(bc['value'])
                elif bc['type'] == 'neumann':
                    if idx == 0:
                        A[idx, 0] = -3 / (2 * dx)
                        A[idx, 1] = 4 / (2 * dx)
                        A[idx, 2] = -1 / (2 * dx)
                    elif idx == n - 1:
                        A[idx, n - 3] = 1 / (2 * dx)
                        A[idx, n - 2] = -4 / (2 * dx)
                        A[idx, n - 1] = 3 / (2 * dx)
                    b[idx] = float(bc['value'])

        try:
            solution = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(A, b, rcond=None)[0]

        return solution


    # def _solve_time_dependent_1d(self, dx):
    #     """Solve time-dependent 1D PDE using explicit finite differences."""
    #     x_vals = self.mesh['x']
    #     t_vals = self.mesh['t']
    #     dt = t_vals[1] - t_vals[0]
    #     nx, nt = len(x_vals), len(t_vals)
        
    #     # Initialize solution array
    #     u = np.zeros((nt, nx))
        
    #     # Set initial condition
    #     if self.initial_condition:
    #         x_sym = self.independent_vars[0]  # Assuming x is first
    #         ic_func = lambdify(x_sym, self.initial_condition, 'numpy')
    #         u[0, :] = ic_func(x_vals)
        
    #     # Time stepping (explicit Euler for simplicity)
    #     for t_idx in range(nt - 1):
    #         for x_idx in range(1, nx - 1):
    #             # Second derivative approximation
    #             u_xx = (u[t_idx, x_idx+1] - 2*u[t_idx, x_idx] + u[t_idx, x_idx-1]) / dx**2
                
    #             # Heat equation: du/dt = α * d²u/dx²
    #             # Simple explicit scheme (assuming heat equation)
    #             alpha = 0.1  # Thermal diffusivity
    #             u[t_idx + 1, x_idx] = u[t_idx, x_idx] + dt * alpha * u_xx
            
    #         # Apply boundary conditions at each time step
    #         for bc in self.boundary_conditions:
    #             if 'x' in bc['location']:
    #                 x_bc = bc['location']['x']
    #                 x_idx = np.argmin(np.abs(x_vals - x_bc))
                    
    #                 if bc['type'] == 'dirichlet':
    #                     # Check if BC depends on time
    #                     if 't' in [str(s) for s in bc['value'].free_symbols]:
    #                         t_sym = symbols('t')
    #                         bc_func = lambdify(t_sym, bc['value'], 'numpy')
    #                         u[t_idx + 1, x_idx] = bc_func(t_vals[t_idx + 1])
    #                     else:
    #                         u[t_idx + 1, x_idx] = float(bc['value'])
        
    #     return u

    def _solve_time_dependent_1d(self, dx):
        x_vals = self.mesh['x']
        t_vals = self.mesh['t']
        dt = t_vals[1] - t_vals[0]
        nx, nt = len(x_vals), len(t_vals)

        u = np.zeros((nt, nx))

        if self.initial_condition:
            x_sym = self.independent_vars[0]
            ic_func = lambdify(x_sym, self.initial_condition, 'numpy')
            u[0, :] = ic_func(x_vals)

        for t_idx in range(nt - 1):
            for x_idx in range(1, nx - 1):
                u_xx = (u[t_idx, x_idx+1] - 2*u[t_idx, x_idx] + u[t_idx, x_idx-1]) / dx**2
                alpha = 0.1
                u[t_idx + 1, x_idx] = u[t_idx, x_idx] + dt * alpha * u_xx

            for bc in self.boundary_conditions:
                if 'x' in bc['location']:
                    x_bc = bc['location']['x']
                    x_idx = np.argmin(np.abs(x_vals - x_bc))

                    if bc['type'] == 'dirichlet':
                        val = sympify(bc['value'])  # <-- convert here safely
                        if 't' in [str(s) for s in val.free_symbols]:
                            t_sym = symbols('t')
                            bc_func = lambdify(t_sym, val, 'numpy')
                            u[t_idx + 1, x_idx] = bc_func(t_vals[t_idx + 1])
                        else:
                            u[t_idx + 1, x_idx] = float(val)

        return u
            
    def _finite_difference_2d(self):
        """Solve 2D PDE using finite differences."""
        x_vals = self.mesh['x']
        y_vals = self.mesh['y']
        dx = x_vals[1] - x_vals[0]
        dy = y_vals[1] - y_vals[0]
        nx, ny = len(x_vals), len(y_vals)
        
        # Create 2D grid
        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        
        # For Laplacian: ∇²u = ∂²u/∂x² + ∂²u/∂y² = 0 (Laplace equation)
        # Convert 2D problem to 1D system
        total_points = nx * ny
        A = np.zeros((total_points, total_points))
        b = np.zeros(total_points)
        
        # Helper function to convert 2D indices to 1D
        def idx(i, j):
            return i * ny + j
        
        # Interior points
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                row = idx(i, j)
                # Central difference for Laplacian
                A[row, idx(i-1, j)] = 1/dx**2  # u[i-1,j]
                A[row, idx(i+1, j)] = 1/dx**2  # u[i+1,j]
                A[row, idx(i, j-1)] = 1/dy**2  # u[i,j-1]
                A[row, idx(i, j+1)] = 1/dy**2  # u[i,j+1]
                A[row, idx(i, j)] = -2/dx**2 - 2/dy**2  # u[i,j]
        
        # Apply boundary conditions
        for bc in self.boundary_conditions:
            if bc['type'] == 'dirichlet':
                # Apply Dirichlet BCs on boundaries
                location = bc['location']
                if 'x' in location:
                    x_bc = location['x']
                    i = np.argmin(np.abs(x_vals - x_bc))
                    for j in range(ny):
                        row = idx(i, j)
                        A[row, :] = 0
                        A[row, row] = 1
                        
                        # Evaluate boundary value
                        if hasattr(bc['value'], 'free_symbols') and bc['value'].free_symbols:
                            x_sym, y_sym = symbols('x y')
                            bc_func = lambdify([x_sym, y_sym], bc['value'], 'numpy')
                            b[row] = bc_func(x_vals[i], y_vals[j])
                        else:
                            b[row] = float(bc['value'])
                            
                if 'y' in location:
                    y_bc = location['y']
                    j = np.argmin(np.abs(y_vals - y_bc))
                    for i in range(nx):
                        row = idx(i, j)
                        A[row, :] = 0
                        A[row, row] = 1
                        
                        # Evaluate boundary value
                        if hasattr(bc['value'], 'free_symbols') and bc['value'].free_symbols:
                            x_sym, y_sym = symbols('x y')
                            bc_func = lambdify([x_sym, y_sym], bc['value'], 'numpy')
                            b[row] = bc_func(x_vals[i], y_vals[j])
                        else:
                            b[row] = float(bc['value'])
        
        # Solve the linear system
        try:
            solution_1d = spsolve(A, b)
        except:
            solution_1d = np.linalg.lstsq(A, b, rcond=None)[0]
            
        # Reshape back to 2D
        solution = solution_1d.reshape((nx, ny))
        return solution
        
    def solve(self, method='finite_difference'):
        """
        Solve the PDE numerically.
        
        Args:
            method: Numerical method to use ('finite_difference')
        """
        if self.mesh is None:
            raise ValueError("Create mesh first using create_mesh()")
            
        print(f"Solving PDE using {method}...")
        
        if method == 'finite_difference':
            if len(self.independent_vars) == 1 or self.time_dependent:
                self.solution = self._finite_difference_1d()
            elif len(self.independent_vars) == 2 and not self.time_dependent:
                self.solution = self._finite_difference_2d()
            else:
                raise ValueError("Unsupported PDE configuration")
        else:
            raise ValueError(f"Method {method} not implemented")
            
        print("PDE solved successfully!")
        return self.solution
        
    def visualize(self, animate=False, save_path=None):
        """
        Visualize the solution.
        
        Args:
            animate: Whether to create animation for time-dependent problems
            save_path: Path to save the plot/animation
        """
        if self.solution is None:
            raise ValueError("Solve the PDE first")
            
        plt.figure(figsize=(12, 8))
        
        if self.time_dependent and len(self.independent_vars) == 2:
            # Time-dependent 1D problem
            x_vals = self.mesh['x']
            t_vals = self.mesh['t']
            
            if animate:
                fig, ax = plt.subplots(figsize=(10, 6))
                line, = ax.plot([], [], 'b-', linewidth=2)
                ax.set_xlim(x_vals.min(), x_vals.max())
                ax.set_ylim(self.solution.min() - 0.1, self.solution.max() + 0.1)
                ax.set_xlabel('x')
                ax.set_ylabel(f'{self.dependent_var}')
                ax.set_title('PDE Solution Evolution')
                ax.grid(True)
                
                def animate_func(frame):
                    line.set_data(x_vals, self.solution[frame, :])
                    ax.set_title(f'PDE Solution at t = {t_vals[frame]:.3f}')
                    return line,
                
                anim = FuncAnimation(fig, animate_func, frames=len(t_vals), 
                                   interval=50, blit=True, repeat=True)
                plt.show()
                
                if save_path:
                    anim.save(save_path, writer='pillow')
            else:
                # Create contour plot
                X, T = np.meshgrid(x_vals, t_vals, indexing='ij')
                plt.contourf(T, X, self.solution.T, levels=50, cmap='viridis')
                plt.colorbar(label=f'{self.dependent_var}')
                plt.xlabel('Time (t)')
                plt.ylabel('Space (x)')
                plt.title('PDE Solution: Space-Time Evolution')
                
        elif len(self.independent_vars) == 2 and not self.time_dependent:
            # 2D steady-state problem
            x_vals = self.mesh['x']
            y_vals = self.mesh['y']
            X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
            
            plt.subplot(1, 2, 1)
            contour = plt.contourf(X, Y, self.solution, levels=50, cmap='viridis')
            plt.colorbar(contour, label=f'{self.dependent_var}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('2D PDE Solution (Contour)')
            
            plt.subplot(1, 2, 2)
            ax = plt.axes(projection='3d')
            surf = ax.plot_surface(X, Y, self.solution, cmap='viridis', alpha=0.9)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel(f'{self.dependent_var}')
            ax.set_title('2D PDE Solution (3D Surface)')
            plt.colorbar(surf, shrink=0.5)
            
        else:
            # 1D steady-state problem
            x_vals = self.mesh['x']
            plt.plot(x_vals, self.solution, 'b-', linewidth=2, label='Numerical Solution')
            plt.xlabel('x')
            plt.ylabel(f'{self.dependent_var}')
            plt.title('1D PDE Solution')
            plt.grid(True)
            plt.legend()
            
        plt.tight_layout()
        if save_path and not animate:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

