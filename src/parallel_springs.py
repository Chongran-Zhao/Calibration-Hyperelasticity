import sympy as sp
import inspect
import random
import numpy as np

class ParallelNetwork:
    """
    A composite material model container representing a parallel network of springs.
    
    Concept:
        Similar to the 'unique_ptr' pattern in C++, this class maintains ownership 
        of multiple material model strategies and manages their lifecycles/parameters.
        It aggregates the energy density from all sub-models.
    
    Physics:
        Corresponds to the isostrain condition where F_total = F_1 = F_2 = ...
        Psi_total = Sum(Psi_i)
    """
    
    def __init__(self):
        # Stores component configuration: [{'func': model_func, 'prefix': 'name', ...}]
        self.components = []
        
        # Combined flat list of parameters for the optimizer
        self.param_names = []
        self.initial_guess = []
        self.bounds = []
        
        # Metadata to masquerade as a standard stretch-based model function
        self.model_type = 'stretch_based'
        self.category = 'composite'
        self.formula = r"\Psi_{total} = \sum_{i=1}^{N} \Psi_{i}"

    def add_model(self, model_func, name_prefix):
        """
        Adds a new parallel branch to the network.
        
        Args:
            model_func: The material model function (from MaterialModels).
            name_prefix (str): Unique identifier for this branch (e.g., 'Matrix', 'Fiber').
        """
        # 1. Inspect original model metadata
        orig_params = getattr(model_func, 'param_names', [])
        orig_guess = getattr(model_func, 'initial_guess', [])
        orig_bounds = getattr(model_func, 'bounds', [])
        
        # 2. Namespace the parameters (e.g., C1 -> Matrix_C1)
        new_params = [f"{name_prefix}_{p}" for p in orig_params]
        
        # 3. Store the component
        self.components.append({
            'func': model_func,
            'prefix': name_prefix,
            'local_params': orig_params,
            'global_params': new_params,
            'orig_type': getattr(model_func, 'model_type', 'stretch_based'),
            'solver': None,
        })
        
        # 4. Update global configuration with PERTURBATION
        self.param_names.extend(new_params)
        
        # --- FIX: Break Symmetry ---
        # If we have multiple branches, slightly perturb the initial guess.
        # This prevents identical models from getting stuck in symmetric optimization paths.
        current_branch_idx = len(self.components)
        
        perturbed_guess = []
        for val in orig_guess:
            # Factor logic:
            # Branch 1: 1.0 (Original)
            # Branch 2: 1.1 (+10%)
            # Branch 3: 0.9 (-10%)
            # Branch 4: 1.2 (+20%) ...
            if current_branch_idx == 1:
                factor = 1.0
            else:
                # Deterministic perturbation based on index to ensure reproducibility
                # Parity check to alternate up/down scaling
                sign = 1.0 if current_branch_idx % 2 == 0 else -1.0
                scale = 0.1 * (current_branch_idx // 2 + 1) # 0.1, 0.2, etc.
                factor = 1.0 + sign * scale
            
            # Apply factor, but ensure we don't accidentally flip signs if not intended
            # (Assuming most params are positive moduli/exponents)
            new_val = val * factor
            
            # Simple safety check: if original was positive, keep it positive
            if val > 0 and new_val <= 0:
                new_val = val * 0.5 
                
            perturbed_guess.append(new_val)

        print(f"  [ParallelNetwork] Added '{name_prefix}' branch ({model_func.__name__}).")
        print(f"    > Initial Guess Perturbed: {orig_guess} -> {[round(x,3) for x in perturbed_guess]}")

        self.initial_guess.extend(perturbed_guess)
        
        if orig_bounds:
            self.bounds.extend(orig_bounds)
        else:
            self.bounds.extend([(None, None)] * len(new_params))

        if getattr(model_func, 'model_type', None) == 'custom':
            self.model_type = 'custom'
        elif self.model_type != 'custom':
            self.model_type = 'stretch_based'

    def __call__(self, lambda_1, lambda_2, lambda_3, params):
        """
        Calculates the total strain energy density.
        Matches the signature required by Kinematics._prepare_stretch_derivatives.
        """
        total_psi = 0
        
        # Pre-calculate Invariants (Assuming Incompressibility J=1)
        # This allows mixing Invariant-based models into this Stretch-based container.
        I1 = lambda_1**2 + lambda_2**2 + lambda_3**2
        # For J=1, I2 = lambda_1^-2 + lambda_2^-2 + lambda_3^-2
        I2 = lambda_1**(-2) + lambda_2**(-2) + lambda_3**(-2)
        
        for comp in self.components:
            model_func = comp['func']
            prefix = comp['prefix']
            
            # Map global params back to local model params
            # e.g. {'Matrix_C1': 10} -> {'C1': 10}
            local_params_dict = {}
            for lp, gp in zip(comp['local_params'], comp['global_params']):
                local_params_dict[lp] = params[gp]
            
            # Dispatch based on the sub-model's type
            if comp['orig_type'] == 'invariant_based':
                # Determine if model needs (I1) or (I1, I2)
                # We use introspection or try/except to handle different signatures
                sig_params = inspect.signature(model_func).parameters
                if 'I2' in sig_params:
                    term = model_func(I1, I2, local_params_dict)
                else:
                    term = model_func(I1, local_params_dict)
            else:
                # Stretch based
                term = model_func(lambda_1, lambda_2, lambda_3, local_params_dict)
                
            total_psi += term
            
        return total_psi

    def custom_pk1(self, F, params):
        """
        Computes total PK1 stress for custom or mixed networks.
        """
        total = np.zeros((3, 3), dtype=float)
        for comp in self.components:
            model_func = comp['func']
            local_params_dict = {}
            for lp, gp in zip(comp['local_params'], comp['global_params']):
                local_params_dict[lp] = params[gp]

            if comp['orig_type'] == 'custom':
                total += model_func.custom_pk1(F, local_params_dict)
            else:
                if comp['solver'] is None:
                    from kinematics import Kinematics
                    comp['solver'] = Kinematics(model_func, comp['local_params'])
                total += comp['solver'].get_1st_PK_stress(F, local_params_dict)
        return total
