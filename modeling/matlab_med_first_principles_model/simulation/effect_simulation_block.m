function ef = effect_simulation_block(ef, ann, varargin)

    iP = inputParser;    
    check_logical = @(x)validateattributes(x, {'logical'},{'scalar'});
    
    addParameter(iP, 'first_effect', false, check_logical)
    addParameter(iP, 'final_effect', false, check_logical)
    addParameter(iP, 'source', "liquid")
    addParameter(iP, 'ef_ref', [])

    parse(iP,varargin{:})
    first_effect = iP.Results.first_effect;
    final_effect = iP.Results.final_effect;
    source = iP.Results.source;
    ef_ref = iP.Results.ef_ref;

    options_fsolve = optimoptions('fsolve','Display','none', 'Algorithm','levenberg-marquardt');

    % Obtain Uef for the input variables
    inputs = [ef.op_time, ef.Ms, ef.Ts_in, ef.Ts_out, ef.Mdest_in, ef.Tdest_in, ...
              ef.Mv_in, ef.Tv_in, ef.Mb_in, ef.Tb_in, ef.Xb_in, ef.Aef];
    ef.Uef = evaluate_trained_ann(inputs, ann);
%     ef.Uef = ef_ref.Uef;
    
    % 1st effect
    if first_effect
        % Cálculos del efecto
        x0 = [ef.Ts_in-5,ef.Mb_in*0.1,ef.Mb_in*0.9,ef.Xb_in*1.1,ef.Ts_in-5];

        if strcmp(source, 'liquid')                
            a = fsolve(@(x) efecto_1_LIQ(x, ef.Uef, ef.Aef, ef.Ms, ef.Ts_in,ef.Ts_out, ef.Tb_in, ef.Mb_in, ef.Xb_in, ef.Cp_s),x0,options_fsolve);

        elseif strcmp(source, 'steam')
            a = fsolve(@(x) efecto_1(x, ef.Uef, ef.Aef, ef.Ms, ef.Ts_in, ef.Tb_in, ef.Mb_in, ef.Xb_in),x0,options_fsolve);
        end
        
        [ef.Tb_out, ef.Mgb, ef.Mb_out, ef.Xb_out, ef.Tv_out] = deal(a(1),a(2),a(3),a(4),a(5));
        ef.Mmix_out = 0;
        ef.Tmix_out = 0;

        % Checks
        % Check that Tb_out < Ts_out (liquid source) or Tb_out < Ts_in (steam source)
        if ef.Tb_out > ef.Ts_out && strcmp(source, 'liquid')
            ME = MException('modelo_simulacion:NonValidTf', ...
                            'Input Tf (%.2f) produced an infeasibility: Tb(1) (%.2f) > Ts_out (%.2f)', Tf, ef.Tb_out, ef.Ts_out);
            throw(ME);
        elseif ef.Tb_out > ef.Ts_in && strcmp(source, 'steam')
            ME = MException('modelo_simulacion:NonValidTf', ...
                            'Input Tf (%.2f) produced an infeasibility: Tb(1) (%.2f) > Ts_in (%.2f)', Tf, ef.Tb_out, ef.Ts_in);
            throw(ME);
        end

    % Every other effect
    else

        % EFFECT MIXER: 
        % vh, delta, d_in, mix_in -> dest_in (M, T)
        % Obtener calores específicos
        ef = ef.calculoScpHeat();
        % Balance de masa:
        ef.Mdest_in = ef.Mvh + ef.Mdelta + ef.Md_in + ef.Mda;
        % Balance de energía:
        if ef.Mdest_in > 0
            ef.Tdest_in = ( ef.Mvh * ef.Tv_in * ef.Cp_ph +...
                            ef.Mdelta * ef.Tv_in * ef.Cp_delta +...
                            ef.Md_in * ef.Td_in * ef.Cp_d +...
                            ef.Mda * ef.Tmix_in * ef.Cp_mix ) / ( ef.Mdest_in * ef.Cp_ph );
        else
            ef.Tdest_in = 0;
        end
        
        % Cálculos del efecto
        x0 = [ef.Mb_in*0.1,ef.Mb_in,ef.Xb_in,ef.Tv_in-2.7,ef.Tb_in-2.7,0,...
              ef.Mb_in,ef.Xb_in,ef.Tb_in-2.7,ef.Mv_in, ef.Mdest_in, 0, ef.Tdest_in, ef.Tdest_in-2.7];
    
        a = fsolve(@(x) efecto_i_flash2(x,ef, ef.Uef), x0, options_fsolve);
    
        [ef.Mgb, ef.Mb_out, ef.Xb_out, ef.Tv_out, ef.Tbb, ...
         ef.Mgf, ef.Mbb, ef.Xbb, ef.Tb_out, ef.Mv_in, ef.Mdest, ...
         ef.Mdest_f, ef.Tdest, ef.Tdest_out] = ...
                    deal(a(1),a(2),a(3),a(4),a(5),a(6),a(7),a(8),a(9),a(10), ...
                         a(11),a(12),a(13),a(14));
    
%         ef = ef.calculoScpHeat; %#ok<*AGROW>
        
        ef.Md_out = ef.Mdest + ef.Mv_in;
        ef.Td_out = ( ef.Mv_in * ef.Tv_in * cpSatLiqTW(ef.Tv_in) + ...
                      ef.Mdest * cpSatLiqTW(ef.Tdest_out) * ef.Tdest_out )...
                    / ( ef.Md_out * ( cpSatLiqTW(ef.Tv_in) + cpSatLiqTW(ef.Tdest_out) )/2 );
    
        if ef.Mda > 0
            ef.Mmix_out = ef.Mdb + ef.Md_out;
            ef.Tmix_out = ( ef.Mdb * ef.Tmix_in * ef.Cp_mix + ef.Md_out ...
                            * ef.Td_out * cpSatLiqTW(ef.Td_out) ) ...
                          / (ef.Mmix_out * ef.Cp_d);
        elseif final_effect
            % MIXER 14
            ef.Mmix_out = ef.Mdb + ef.Md_out; 
            ef.Tmix_out = ( ef.Mdb * ef.Tmix_in * cpSatLiqTW(ef.Tmix_in) + ...
                            ef.Md_out * ef.Td_out * cpSatLiqTW(ef.Td_out) )...
                            / (ef.Mdb * cpSatLiqTW(ef.Tmix_in));
            ef.Cp_mix = cpSatLiqTW(ef.Tmix_out);
        else
            ef.Mmix_out = 0;
            ef.Tmix_out = 0;
        end
    
        % Checks
        % Comprobar balance de materia en efecto
        if abs(ef.Mgf + ef.Mgb + ef.Mb_out - ef.Mb_in) > 1e-4
            warning("No se cumple balance de materia: Mb_in != Mgb + Mgf + Mb_out (%.2f != %.2f)",...
                    ef.Mb_in*3600, (ef.Mgf+ef.Mgb+ef.Mb_out)*3600)
        end
    end
    % Obtain energy input for each source
    ef = ef.estimacionAportes2;
    
    % Common outputs
    ef.Mv_out = ef.Mgb + ef.Mgf;

end