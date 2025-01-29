
def fitness(self, x):
    """ Legacy method, when the structure of the decision variables was:
        x = [x1,0, x2,0, ..., xN,0, x1,1, x2,1, ..., xN,1, ..., x1,T, x2,T, ..., xN,T]
    """
    for step_opt_idx in range(self.n_horizon):
        span: tuple[int, int] = (step_opt_idx*self.n_dec_vars, (step_opt_idx+1)*self.n_dec_vars)
        dec_vars: DecisionVariables = DecisionVariables(**{input_id: input_value for input_id, input_value in zip(self.dec_var_ids, 
                                                                                                                x[span[0]:span[1]])})
        for step_mod_idx in range(self.sample_time/self.model.sample_time):
            model.step(
                # Decision variables
                ## Thermal storage
                qts_src=dec_vars.qts_src * dec_vars.ts_active,
                
                ## Solar field
                qsf=dec_vars.qsf * dec_vars.sf_active,
                
                ## MED
                qmed_s=dec_vars.qmed_s * dec_vars.med_active,
                qmed_f=dec_vars.qmed_f * dec_vars.med_active,
                Tmed_s_in=dec_vars.Tmed_s_in,
                Tmed_c_out=dec_vars.Tmed_c_out,
                med_vacuum_state=dec_vars.med_vac_state,
                
                ## Environment
                # **self.env_vars.model_dump_at_index(step_mod_idx)
                I=self.env_vars.I[step_mod_idx]
            )
            benefit[step_opt_idx] += model.evaluate_fitness_function(
                cost_e=self.env_vars.cost_e[step_mod_idx],
                cost_w=self.env_vars.cost_w[step_mod_idx]
            )
        
        # Input values might not been valid and modified by the model, retrieve
        # them and return the difference as inequality constraints, is it needed?
        # We can just re-run the evaluation one last time to retrieve the validated
        # inputs from the final solution
        ci: tuple[float, ...] = (abs(float(getattr(model, var_id) - x[span[0]:span[1]])) for var_id in self.dec_var_ids)
        
        return np.sum(benefit), ci

def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """ Legacy method, when the structure of the decision variables was:
            x = [x1,0, x2,0, ..., xN,0, x1,1, x2,1, ..., xN,1, ..., x1,T, x2,T, ..., xN,T]
        """
        """This method will return the box-bounds of the problem. 
        Infinities in the bounds are allowed.

        Returns:
            tuple[np.ndarray, np.ndarray]: (lower bounds, upper bounds)
        """
        
        """ TODO: Refactor after changing the decision variables structure """
        
        # The order of elements in the bounds should match the order of the variables in the decision vector (`dec_var_ids`)
        
        # To simplify, use a 2D array with shape (n_horizon, n_dec_vars) to set bounds, later reshape it to (n_horizon x n_dec_vars)
        box_bounds_lower: np.ndarray[float | int] = np.full((self.n_horizon, self.n_dec_vars), np.nan, dtype=float) # (n steps, n inputs)
        box_bounds_upper: np.ndarray[float | int] = np.full((self.n_horizon, self.n_dec_vars), np.nan, dtype=float) # (n steps, n inputs)
        integer_dec_vars_mapping: dict[str, np.ndarray[list[int]]] = {var_id: np.full((self.n_horizon, ), list[int], dtype=object) 
                                                                for var_id in self.dec_var_int_ids}
        
        for initial_state, fsm_data in zip([self.model._sf_ts_fsm.state, self.model._med_fsm.state],
                                           [self.fsm_sfts_data, self.fsm_med_data]):
            
            paths_df = fsm_data.paths_df
            valid_inputs = fsm_data.valid_inputs
            input_ids: list[str] = fsm_data.metadata["input_ids"] 
            
            n_dec_vars: int = len(input_ids)
            
            # Handle logical / integer variables for both FSMs
            # Extract indexes of possible paths from initial state
            paths_from_state_idxs: np.ndarray = paths_df[paths_df["0"] == initial_state.value].index.to_numpy() # dim: (n paths, )
            # Get valid inputs from initial states using those indexes
            valid_inputs_from_state: np.ndarray = valid_inputs[paths_from_state_idxs] # dim: (n paths, n horizon, n inputs)

            # Get the unique discrete values for each input
            # Initialize an empty array to store the unique elements
            unique_elements: np.ndarray = np.empty((self.n_horizon, n_dec_vars), dtype=object) # dim: (n horizon, n inputs, n unique inputs)

            # Find unique valid inputs for step, from all possible paths
            for step_idx in range(self.n_horizon): # For every step
                for input_idx in range(n_dec_vars): # For every input
                    unique_elements[step_idx, input_idx] = np.unique(valid_inputs_from_state[:, step_idx, input_idx])
            # print(unique_elements)
            discrete_bounds = unique_elements
            
            # Map the discrete values to box bounds
            for step_idx in range(self.n_horizon): # Step
                for input_idx, fsm_input_id in enumerate(input_ids): # Input
                    optim_input_id = VarIdsOptimToFsmsMapping(fsm_input_id).name
                    # Update mapping
                    integer_dec_vars_mapping[optim_input_id][step_idx] = discrete_bounds[step_idx, input_idx]
                    # Update bounds
                    input_idx_in_dec_vars = self.dec_var_ids.index(optim_input_id)
                    box_bounds_upper[step_idx, input_idx_in_dec_vars] = len(discrete_bounds[step_idx, input_idx])-1
                    box_bounds_lower[step_idx, input_idx_in_dec_vars] = 0

        # Real variables bounds
        # Done manually for now
        # ['qsf', 'qts_src', 'qmed_s', 'qmed_f', 'Tmed_s_in', 'Tmed_c_out']
        def set_real_var_bounds(var_id: str, lower_limit: float | np.ndarray[float], 
                                upper_limit: float | np.ndarray[float], aux_logical_var_id: str  = None) -> None:
            """
            Set the bounds for a real variable in the decision vector
            No need to pass the bounds arrays or return them, as they are modified in place (mutable)
            """
            input_idx_in_dec_vars = self.dec_var_ids.index(var_id)
            if aux_logical_var_id is not None:
                aux_logical_input_idx_in_dec_vars = self.dec_var_ids.index(aux_logical_var_id)
                upper_value: np.ndarray[float] = upper_limit * box_bounds_upper[:, aux_logical_input_idx_in_dec_vars]
                lower_value: np.ndarray[float] = lower_limit * box_bounds_upper[:, aux_logical_input_idx_in_dec_vars]
            else:
                upper_value: np.ndarray[float] = upper_limit * np.ones((self.n_horizon, ))
                lower_value: np.ndarray[float] = lower_limit * np.ones((self.n_horizon, ))
            
            box_bounds_upper[:, input_idx_in_dec_vars] = upper_value
            box_bounds_lower[:, input_idx_in_dec_vars] = lower_value
        
        # Solar field
        set_real_var_bounds(
            var_id = 'qsf', 
            lower_limit = self.model.fixed_model_params.sf.qsf_min, 
            upper_limit = self.model.fixed_model_params.sf.qsf_max, 
            aux_logical_var_id = 'sf_active'
        )
        # Thermal storage
        set_real_var_bounds(
            var_id = 'qts_src', 
            lower_limit = self.model.fixed_model_params.ts.qts_src_min, 
            upper_limit = self.model.fixed_model_params.ts.qts_src_max, 
            aux_logical_var_id = 'ts_active'
        )
        # MED
        set_real_var_bounds(
            var_id = 'Tmed_s_in', 
            lower_limit = self.model.fixed_model_params.med.Tmed_s_min, 
            upper_limit = self.model.fixed_model_params.med.Tmed_s_max, 
            aux_logical_var_id = 'med_active'
        )
        set_real_var_bounds(
            var_id = 'Tmed_c_out', 
            lower_limit = self.env_vars.Tmed_c_in, 
            upper_limit = self.env_vars.Tmed_c_in+10, # A temperature delta of over 10ºC is unfeasible for the condenser
            aux_logical_var_id = 'med_active'
        )
        for var_id in ['qmed_s', 'qmed_f']:
            set_real_var_bounds(
                var_id = var_id, 
                lower_limit = getattr(self.model.fixed_model_params.med, f"{var_id}_min"), 
                upper_limit = getattr(self.model.fixed_model_params.med, f"{var_id}_max"), 
                aux_logical_var_id = 'med_active'
            )

        # Finally, the output shape for the bounds should be (n_horizon x n_dec_vars), reshape
        box_bounds_lower = box_bounds_lower.reshape(-1)
        box_bounds_upper = box_bounds_upper.reshape(-1)
        
        self.integer_dec_vars_mapping = integer_dec_vars_mapping
        
        # Validate that all decision variables bounds have been set
        # TODO: Should be commented out in production
        # assert np.isnan(box_bounds_upper).any(), "Some upper bounds were not set"
        # assert np.isnan(box_bounds_lower).any(), "Some lower bounds were not set"

        return box_bounds_lower, box_bounds_upper