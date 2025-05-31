# Test Classes and Fixtures
import numpy as np
# from pytest import Function
from sympy import Function
import pytest
import sympy as sp
from sympy import symbols, diff, sin, pi
from pde_solver import SymbolicNumericPDESolver
from sympy import S

# Test Classes and Fixtures
class TestSymbolicNumericPDESolver:
    """Comprehensive test suite for the PDE solver."""
    
    @pytest.fixture
    def solver(self):
        """Create a fresh solver instance for each test."""
        return SymbolicNumericPDESolver()
    
    @pytest.fixture
    def basic_symbols(self):
        """Common symbolic variables for testing."""
        return {
            'x': symbols('x'),
            'y': symbols('y'),
            't': symbols('t'),
            'u': symbols('u')
        }
    
    # Basic Initialization Tests
    def test_solver_initialization(self, solver):
        """Test that solver initializes with correct default values."""
        assert solver.pde is None
        assert solver.domain is None
        assert solver.boundary_conditions == []
        assert solver.initial_condition is None
        assert solver.solution is None
        assert solver.mesh is None
        assert solver.time_dependent is False
    
    # PDE Definition Tests
    def test_define_1d_steady_pde(self, solver, basic_symbols):
        """Test defining a 1D steady-state PDE."""
        x, u = basic_symbols['x'], basic_symbols['u']
        u_func = Function('u')(x)
        pde_expr = diff(u_func, x, x) + 1  # Poisson equation
        
        solver.define_pde(pde_expr, u, [x], {'x': (0, 1)})
        
        assert solver.pde == pde_expr
        assert solver.dependent_var == u
        assert solver.independent_vars == [x]
        assert solver.domain_bounds == {'x': (0, 1)}
        assert solver.time_dependent is False
    
    def test_define_1d_time_dependent_pde(self, solver, basic_symbols):
        """Test defining a 1D time-dependent PDE."""
        x, t, u = basic_symbols['x'], basic_symbols['t'], basic_symbols['u']
        u_func = Function('u')(x, t)
        pde_expr = diff(u_func, t) - 0.1 * diff(u_func, x, x)  # Heat equation
        
        solver.define_pde(pde_expr, u, [x, t], {'x': (0, 1), 't': (0, 0.5)})
        
        assert solver.pde == pde_expr
        assert solver.time_dependent is True
    
    def test_define_2d_steady_pde(self, solver, basic_symbols):
        """Test defining a 2D steady-state PDE."""
        x, y, u = basic_symbols['x'], basic_symbols['y'], basic_symbols['u']
        u_func = Function('u')(x, y)
        pde_expr = diff(u_func, x, x) + diff(u_func, y, y)  # Laplace equation
        
        solver.define_pde(pde_expr, u, [x, y], {'x': (0, 1), 'y': (0, 1)})
        
        assert solver.pde == pde_expr
        assert solver.time_dependent is False
        assert len(solver.independent_vars) == 2
    
    # Boundary Condition Tests
    def test_add_dirichlet_boundary_condition(self, solver):
        """Test adding Dirichlet boundary conditions."""
        solver.add_boundary_condition('dirichlet', {'x': 0}, 5.0)
        
        assert len(solver.boundary_conditions) == 1
        bc = solver.boundary_conditions[0]
        assert bc['type'] == 'dirichlet'
        assert bc['location'] == {'x': 0}
        assert bc['value'] == 5.0
        assert bc['variable'] is None
    
    def test_add_neumann_boundary_condition(self, solver):
        """Test adding Neumann boundary conditions."""
        solver.add_boundary_condition('neumann', {'x': 1}, 2.0, variable='x')
        
        assert len(solver.boundary_conditions) == 1
        bc = solver.boundary_conditions[0]
        assert bc['type'] == 'neumann'
        assert bc['location'] == {'x': 1}
        assert bc['value'] == 2.0
        assert bc['variable'] == 'x'
    
    def test_add_multiple_boundary_conditions(self, solver):
        """Test adding multiple boundary conditions."""
        solver.add_boundary_condition('dirichlet', {'x': 0}, 0)
        solver.add_boundary_condition('dirichlet', {'x': 1}, 1)
        solver.add_boundary_condition('neumann', {'y': 0}, 0, variable='y')
        
        assert len(solver.boundary_conditions) == 3
        assert solver.boundary_conditions[0]['type'] == 'dirichlet'
        assert solver.boundary_conditions[1]['type'] == 'dirichlet'
        assert solver.boundary_conditions[2]['type'] == 'neumann'
    
    def test_symbolic_boundary_condition(self, solver, basic_symbols):
        """Test boundary conditions with symbolic expressions."""
        x = basic_symbols['x']
        solver.add_boundary_condition('dirichlet', {'y': 0}, sin(pi*x))
        
        bc = solver.boundary_conditions[0]
        assert bc['value'] == sp.sin(pi*x)
    
    # Initial Condition Tests
    def test_set_initial_condition(self, solver, basic_symbols):
        """Test setting initial conditions."""
        x = basic_symbols['x']
        ic_expr = sin(pi*x)
        solver.set_initial_condition(ic_expr)
        
        assert solver.initial_condition == ic_expr
    
    # Mesh Creation Tests
    def test_create_1d_mesh(self, solver, basic_symbols):
        """Test creating 1D mesh."""
        x, u = basic_symbols['x'], basic_symbols['u']
        u_func = Function('u')(x)
        pde_expr = diff(u_func, x, x)
        
        solver.define_pde(pde_expr, u, [x], {'x': (0, 1)})
        solver.create_mesh({'x': 21})
        
        assert 'x' in solver.mesh
        assert len(solver.mesh['x']) == 21
        assert solver.mesh['x'][0] == 0.0
        assert solver.mesh['x'][-1] == 1.0
        np.testing.assert_allclose(solver.mesh['x'], np.linspace(0, 1, 21))
    
    def test_create_2d_mesh(self, solver, basic_symbols):
        """Test creating 2D mesh."""
        x, y, u = basic_symbols['x'], basic_symbols['y'], basic_symbols['u']
        u_func = Function('u')(x, y)
        pde_expr = diff(u_func, x, x) + diff(u_func, y, y)
        
        solver.define_pde(pde_expr, u, [x, y], {'x': (0, 1), 'y': (0, 2)})
        solver.create_mesh({'x': 11, 'y': 21})
        
        assert 'x' in solver.mesh
        assert 'y' in solver.mesh
        assert len(solver.mesh['x']) == 11
        assert len(solver.mesh['y']) == 21
        assert solver.mesh['x'][0] == 0.0
        assert solver.mesh['x'][-1] == 1.0
        assert solver.mesh['y'][0] == 0.0
        assert solver.mesh['y'][-1] == 2.0
    
    def test_create_time_dependent_mesh(self, solver, basic_symbols):
        """Test creating mesh for time-dependent problems."""
        x, t, u = basic_symbols['x'], basic_symbols['t'], basic_symbols['u']
        u_func = Function('u')(x, t)
        pde_expr = diff(u_func, t) - diff(u_func, x, x)
        
        solver.define_pde(pde_expr, u, [x, t], {'x': (0, 1), 't': (0, 0.1)})
        solver.create_mesh({'x': 21, 't': 11})
        
        assert 'x' in solver.mesh
        assert 't' in solver.mesh
        assert len(solver.mesh['x']) == 21
        assert len(solver.mesh['t']) == 11
    
    # Solution Tests
    def test_solve_without_mesh_raises_error(self, solver, basic_symbols):
        """Test that solving without creating mesh raises appropriate error."""
        x, u = basic_symbols['x'], basic_symbols['u']
        u_func = Function('u')(x)
        pde_expr = diff(u_func, x, x)
        
        solver.define_pde(pde_expr, u, [x], {'x': (0, 1)})
        
        with pytest.raises(ValueError, match="Create mesh first"):
            solver.solve()
    
    # def test_solve_1d_steady_state_poisson(self, solver, basic_symbols):
    #     """Test solving 1D Poisson equation with known analytical solution."""
    #     x, u = basic_symbols['x'], basic_symbols['u']
    #     u_func = Function('u')(x)
    #     # d²u/dx² = -2, u(0) = 0, u(1) = 0
    #     # Analytical solution: u(x) = x(1-x)
    #     pde_expr = diff(u_func, x, x) + 2
        
    #     solver.define_pde(pde_expr, u, [x], {'x': (0, 1)})
    #     solver.add_boundary_condition('dirichlet', {'x': 0}, 0)
    #     solver.add_boundary_condition('dirichlet', {'x': 1}, 0)
    #     solver.create_mesh({'x': 21})
        
    #     solution = solver.solve()
        
    #     # Check solution properties
    #     assert solution is not None
    #     assert len(solution) == 21
    #     assert isinstance(solution, np.ndarray)
        
    #     # Check boundary conditions are satisfied
    #     assert abs(solution[0] - 0) < 1e-10  # u(0) = 0
    #     assert abs(solution[-1] - 0) < 1e-10  # u(1) = 0
        
    #     # Check against analytical solution at middle point
    #     x_vals = solver.mesh['x']
    #     x_mid = x_vals[len(x_vals)//2]
    #     analytical_mid = x_mid * (1 - x_mid)
    #     numerical_mid = solution[len(x_vals)//2]
    #     assert abs(numerical_mid - analytical_mid) < 0.01  # Should be close
    

    def test_solve_1d_steady_state_poisson(self, solver, basic_symbols):
        x, u = basic_symbols['x'], basic_symbols['u']
        
        # Use symbolic u, not AppliedUndef
        pde_expr = diff(u, x, x) + 2  # u'' + 2 = 0
        
        solver.define_pde(pde_expr, u, [x], {'x': (0, 1)})
        # solver.add_boundary_condition('dirichlet', {'x': 0}, 0)
        # solver.add_boundary_condition('dirichlet', {'x': 1}, 0)
        
        solver.add_boundary_condition('dirichlet', {'x': 0}, S(0))
        solver.add_boundary_condition('dirichlet', {'x': 1}, S(0))
        solver.create_mesh({'x': 21})
        
        solution = solver.solve()

        assert solution is not None
        assert len(solution) == 21
        assert isinstance(solution, np.ndarray)

        # Check boundary conditions
        assert abs(solution[0] - 0) < 1e-10
        assert abs(solution[-1] - 0) < 1e-10

        # Compare numerical solution to analytical u(x) = x(1 - x)
        x_vals = solver.mesh['x']
        x_mid = x_vals[len(x_vals)//2]
        analytical_mid = x_mid * (1 - x_mid)
        numerical_mid = solution[len(x_vals)//2]
        assert abs(numerical_mid - analytical_mid) < 0.01


    # def test_solve_1d_with_neumann_bc(self, solver, basic_symbols):
    #     """Test solving 1D problem with Neumann boundary condition."""
    #     x, u = basic_symbols['x'], basic_symbols['u']
    #     u_func = Function('u')(x)
    #     pde_expr = diff(u_func, x, x) + 1  # d²u/dx² = -1
        
    #     solver.define_pde(pde_expr, u, [x], {'x': (0, 1)})
    #     solver.add_boundary_condition('dirichlet', {'x': 0}, 0)
    #     solver.add_boundary_condition('neumann', {'x': 1}, 0, variable='x')
    #     solver.create_mesh({'x': 21})
        
    #     solution = solver.solve()
        
    #     assert solution is not None
    #     assert len(solution) == 21
    #     assert abs(solution[0] - 0) < 1e-10  # Dirichlet BC at x=0
    
    def test_solve_1d_with_neumann_bc(self, solver, basic_symbols):
        """Test solving 1D problem with Neumann boundary condition."""
        x, u = basic_symbols['x'], basic_symbols['u']
        
        # Use `u` as symbolic dependent variable
        pde_expr = diff(u, x, x) + 1  # ∂²u/∂x² = -1

        solver.define_pde(pde_expr, u, [x], {'x': (0, 1)})
        solver.add_boundary_condition('dirichlet', {'x': 0}, 0)
        solver.add_boundary_condition('neumann', {'x': 1}, 0, variable='x')
        solver.create_mesh({'x': 21})

        solution = solver.solve()

        assert solution is not None
        assert len(solution) == 21
        assert abs(solution[0] - 0) < 1e-10  # Dirichlet BC at x=0


    # def test_solve_2d_laplace_equation_notuse(self, solver, basic_symbols):
    #     """Test solving 2D Laplace equation."""
    #     x, y, u = basic_symbols['x'], basic_symbols['y'], basic_symbols['u']
    #     u_func = Function('u')(x, y)
    #     pde_expr = diff(u_func, x, x) + diff(u_func, y, y)  # ∇²u = 0
        
    #     solver.define_pde(pde_expr, u, [x, y], {'x': (0, 1), 'y': (0, 1)})
    #     solver.add_boundary_condition('dirichlet', {'x': 0}, 0)
    #     solver.add_boundary_condition('dirichlet', {'x': 1}, 0)
    #     solver.add_boundary_condition('dirichlet', {'y': 0}, 1)
    #     solver.add_boundary_condition('dirichlet', {'y': 1}, 0)
    #     solver.create_mesh({'x': 11, 'y': 11})
        
    #     solution = solver.solve()
        
    #     assert solution is not None
    #     assert solution.shape == (11, 11)
        
    #     # Test boundary conditions
    #     assert np.allclose(solution[0, :], 0, atol=1e-10)  # x=0 boundary
    #     assert np.allclose(solution[-1, :], 0, atol=1e-10)  # x=1 boundary
    #     assert np.allclose(solution[:, 0], 1, atol=1e-10)  # y=0 boundary
    #     assert np.allclose(solution[:, -1], 0, atol=1e-10)  # y=1 boundary
    
    # def test_solve_2d_laplace_equation(self, solver, basic_symbols): 
    #     """Test solving 2D Laplace equation."""
    #     x, y, u = basic_symbols['x'], basic_symbols['y'], basic_symbols['u']
        
    #     # Use symbolic form
    #     pde_expr = diff(u, x, x) + diff(u, y, y)  # ∇²u = 0

    #     solver.define_pde(pde_expr, u, [x, y], {'x': (0, 1), 'y': (0, 1)})
    #     solver.add_boundary_condition('dirichlet', {'x': 0}, 0)
    #     solver.add_boundary_condition('dirichlet', {'x': 1}, 0)
    #     solver.add_boundary_condition('dirichlet', {'y': 0}, 1)
    #     solver.add_boundary_condition('dirichlet', {'y': 1}, 0)
    #     solver.create_mesh({'x': 11, 'y': 11})

    #     solution = solver.solve()

    #     assert solution is not None
    #     assert solution.shape == (11, 11)
    #     assert np.allclose(solution[0, :], 0, atol=1e-10)  # x=0
    #     assert np.allclose(solution[-1, :], 0, atol=1e-10)  # x=1
    #     assert np.allclose(solution[:, 0], 1, atol=1e-10)  # y=0
    #     assert np.allclose(solution[:, -1], 0, atol=1e-10)  # y=1


    def test_solve_time_dependent_heat_equation(self, solver, basic_symbols):
        """Test solving 1D heat equation."""
        x, t, u = basic_symbols['x'], basic_symbols['t'], basic_symbols['u']
        u_func = Function('u')(x, t)
        # Heat equation: ∂u/∂t = α∇²u
        alpha = 0.1
        pde_expr = diff(u_func, t) - alpha * diff(u_func, x, x)
        
        solver.define_pde(pde_expr, u, [x, t], {'x': (0, 1), 't': (0, 0.1)})
        solver.add_boundary_condition('dirichlet', {'x': 0}, 0)
        solver.add_boundary_condition('dirichlet', {'x': 1}, 0)
        solver.set_initial_condition(sin(pi*symbols('x')))
        solver.create_mesh({'x': 21, 't': 11})
        
        solution = solver.solve()
        
        assert solution is not None
        assert solution.shape == (11, 21)  # (time_steps, space_points)
        
        # Check boundary conditions at all time steps
        assert np.allclose(solution[:, 0], 0, atol=1e-10)  # x=0 boundary
        assert np.allclose(solution[:, -1], 0, atol=1e-10)  # x=1 boundary
        
        # Check that solution decays over time (heat dissipation)
        initial_max = np.max(np.abs(solution[0, :]))
        final_max = np.max(np.abs(solution[-1, :]))
        assert final_max < initial_max  # Solution should decay
    
    # def test_solve_time_dependent_heat_equation(self, solver, basic_symbols):
    #     """Test solving 1D heat equation."""
    #     x, t, u = basic_symbols['x'], basic_symbols['t'], basic_symbols['u']
    #     alpha = 0.1

    #     pde_expr = diff(u, t) - alpha * diff(u, x, x)
        
    #     solver.define_pde(pde_expr, u, [x, t], {'x': (0, 1), 't': (0, 0.1)})
    #     solver.add_boundary_condition('dirichlet', {'x': 0}, S(0))
    #     solver.add_boundary_condition('dirichlet', {'x': 1}, S(0))
    #     solver.set_initial_condition(sin(pi * x))
    #     solver.create_mesh({'x': 21, 't': 11})

    #     solution = solver.solve()

    #     assert solution is not None
    #     assert solution.shape == (11, 21)
    #     assert np.allclose(solution[:, 0], 0, atol=1e-10)
    #     assert np.allclose(solution[:, -1], 0, atol=1e-10)

    #     initial_max = np.max(np.abs(solution[0, :]))
    #     final_max = np.max(np.abs(solution[-1, :]))
    #     assert final_max < initial_max


    # # Error Handling Tests
    def test_invalid_pde_definition(self, solver, basic_symbols):
        """Test handling of invalid PDE definitions."""
        x, u = basic_symbols['x'], basic_symbols['u']
        
        # Test with inconsistent variables
        with pytest.raises(ValueError):
            solver.define_pde(x + 1, u, [symbols('y')], {'y': (0, 1)})
    
    def test_invalid_boundary_condition_type(self, solver):
        """Test handling of invalid boundary condition types."""
        with pytest.raises(ValueError, match="Unsupported boundary condition"):
            solver.add_boundary_condition('robin', {'x': 0}, 1.0)
    
    def test_incompatible_mesh_dimensions(self, solver, basic_symbols):
        """Test error when mesh dimensions don't match PDE."""
        x, u = basic_symbols['x'], basic_symbols['u']
        u_func = Function('u')(x)
        pde_expr = diff(u_func, x, x)
        
        solver.define_pde(pde_expr, u, [x], {'x': (0, 1)})
        
        with pytest.raises(ValueError):
            solver.create_mesh({'x': 21, 'y': 11})  # Extra dimension
    
    def test_missing_boundary_conditions(self, solver, basic_symbols):
        """Test behavior when boundary conditions are missing."""
        x, u = basic_symbols['x'], basic_symbols['u']
        u_func = Function('u')(x)
        pde_expr = diff(u_func, x, x) + 1
        
        solver.define_pde(pde_expr, u, [x], {'x': (0, 1)})
        solver.create_mesh({'x': 21})
        
        # Should warn or handle gracefully
        solution = solver.solve()
        assert solution is not None  # Should still produce some result
    
    # # Utility and Property Tests
    # def test_get_solution_at_point(self, solver, basic_symbols):
    #     """Test getting solution value at specific points."""
    #     x, u = basic_symbols['x'], basic_symbols['u']
    #     u_func = Function('u')(x)
    #     pde_expr = diff(u_func, x, x) + 2
        
    #     solver.define_pde(pde_expr, u, [x], {'x': (0, 1)})
    #     solver.add_boundary_condition('dirichlet', {'x': 0}, 0)
    #     solver.add_boundary_condition('dirichlet', {'x': 1}, 0)
    #     solver.create_mesh({'x': 21})
    #     solution = solver.solve()
        
    #     # Test interpolation at arbitrary point
    #     value_at_half = solver.get_solution_at_point({'x': 0.5})
    #     assert isinstance(value_at_half, float)
    #     assert value_at_half > 0  # Should be positive for this problem
    
    # def test_solution_convergence(self, solver, basic_symbols):
    #     """Test solution convergence with mesh refinement."""
    #     x, u = basic_symbols['x'], basic_symbols['u']
    #     u_func = Function('u')(x)
    #     pde_expr = diff(u_func, x, x) + 2
        
    #     solver.define_pde(pde_expr, u, [x], {'x': (0, 1)})
    #     solver.add_boundary_condition('dirichlet', {'x': 0}, 0)
    #     solver.add_boundary_condition('dirichlet', {'x': 1}, 0)
        
    #     # Solve with coarse mesh
    #     solver.create_mesh({'x': 11})
    #     solution_coarse = solver.solve()
        
    #     # Solve with fine mesh
    #     solver.create_mesh({'x': 21})
    #     solution_fine = solver.solve()
        
    #     # Fine solution should be more accurate
    #     x_vals_coarse = solver.mesh['x']
    #     analytical_coarse = x_vals_coarse * (1 - x_vals_coarse)
    #     error_coarse = np.max(np.abs(solution_coarse - analytical_coarse))
        
    #     solver.create_mesh({'x': 21})
    #     solution_fine = solver.solve()
    #     x_vals_fine = solver.mesh['x']
    #     analytical_fine = x_vals_fine * (1 - x_vals_fine)
    #     error_fine = np.max(np.abs(solution_fine - analytical_fine))
        
    #     assert error_fine < error_coarse  # Finer mesh should have smaller error
    
    # def test_solver_reset(self, solver, basic_symbols):
    #     """Test resetting solver state."""
    #     x, u = basic_symbols['x'], basic_symbols['u']
    #     u_func = Function('u')(x)
    #     pde_expr = diff(u_func, x, x)
        
    #     # Set up solver
    #     solver.define_pde(pde_expr, u, [x], {'x': (0, 1)})
    #     solver.add_boundary_condition('dirichlet', {'x': 0}, 0)
    #     solver.create_mesh({'x': 21})
        
    #     # Reset and check
    #     solver.reset()
    #     assert solver.pde is None
    #     assert solver.boundary_conditions == []
    #     assert solver.mesh is None
    #     assert solver.solution is None
    
    # def test_multiple_pde_types(self, solver, basic_symbols):
    #     """Test solver with different types of PDEs."""
    #     x, y, t, u = basic_symbols['x'], basic_symbols['y'], basic_symbols['t'], basic_symbols['u']
        
    #     # Test wave equation: ∂²u/∂t² = c²∇²u
    #     u_func = Function('u')(x, t)
    #     c = 1.0
    #     wave_pde = diff(u_func, t, t) - c**2 * diff(u_func, x, x)
        
    #     solver.define_pde(wave_pde, u, [x, t], {'x': (0, 1), 't': (0, 1)})
    #     assert solver.time_dependent is True
        
    #     # Reset and test diffusion equation: ∂u/∂t = D∇²u
    #     solver.reset()
    #     u_func = Function('u')(x, y, t)
    #     D = 0.1
    #     diffusion_pde = diff(u_func, t) - D * (diff(u_func, x, x) + diff(u_func, y, y))
        
    #     solver.define_pde(diffusion_pde, u, [x, y, t], {'x': (0, 1), 'y': (0, 1), 't': (0, 0.5)})
    #     assert solver.time_dependent is True
    #     assert len(solver.independent_vars) == 3
    
    # def test_symbolic_coefficients(self, solver, basic_symbols):
    #     """Test PDEs with symbolic coefficients."""
    #     x, u = basic_symbols['x'], basic_symbols['u']
    #     u_func = Function('u')(x)
    #     k = symbols('k', positive=True)
        
    #     # Variable coefficient PDE: d/dx(k(x) du/dx) + f(x) = 0
    #     pde_expr = diff(k*x * diff(u_func, x), x) + sin(pi*x)
        
    #     solver.define_pde(pde_expr, u, [x], {'x': (0, 1)})
    #     solver.add_boundary_condition('dirichlet', {'x': 0}, 0)
    #     solver.add_boundary_condition('dirichlet', {'x': 1}, 0)
    #     solver.create_mesh({'x': 21})
        
    #     # Should handle symbolic coefficients
    #     solution = solver.solve(parameters={'k': 1.0})
    #     assert solution is not None
    
    # # Performance and Memory Tests
    # def test_large_mesh_performance(self, solver, basic_symbols):
    #     """Test solver performance with larger meshes."""
    #     x, u = basic_symbols['x'], basic_symbols['u']
    #     u_func = Function('u')(x)
    #     pde_expr = diff(u_func, x, x) + 1
        
    #     solver.define_pde(pde_expr, u, [x], {'x': (0, 1)})
    #     solver.add_boundary_condition('dirichlet', {'x': 0}, 0)
    #     solver.add_boundary_condition('dirichlet', {'x': 1}, 0)
    #     solver.create_mesh({'x': 101})  # Larger mesh
        
    #     import time
    #     start_time = time.time()
    #     solution = solver.solve()
    #     solve_time = time.time() - start_time
        
    #     assert solution is not None
    #     assert len(solution) == 101
    #     assert solve_time < 10.0  # Should solve reasonably quickly
    
    # def test_memory_usage(self, solver, basic_symbols):
    #     """Test that solver doesn't leak memory excessively."""
    #     x, u = basic_symbols['x'], basic_symbols['u']
    #     u_func = Function('u')(x)
    #     pde_expr = diff(u_func, x, x) + 1
        
    #     solver.define_pde(pde_expr, u, [x], {'x': (0, 1)})
    #     solver.add_boundary_condition('dirichlet', {'x': 0}, 0)
    #     solver.add_boundary_condition('dirichlet', {'x': 1}, 0)
        
    #     # Solve multiple times and check memory doesn't grow excessively
    #     import gc
    #     for i in range(5):
    #         solver.create_mesh({'x': 51})
    #         solution = solver.solve()
    #         del solution
    #         gc.collect()
        
    #     # Test passes if no memory errors occur
    #     assert True
    # """Comprehensive test suite for the PDE solver."""
   
    