function [ef, ph, out] = simulation_model(pt, varargin)

    iP = inputParser;
    
    check_logical = @(x)validateattributes(x, {'logical'},{'scalar'});
    
    addParameter(iP, 'gfc', false, check_logical)
    addParameter(iP, 'log', true, check_logical)
    addParameter(iP, 'debug_pt',  [])
    addParameter(iP, 'ann_deltaMv', [])
    addParameter(iP, 'Tf', [])
    addParameter(iP, 'save_results', false, check_logical)
    addParameter(iP, 'ptop_idx', [])
    addParameter(iP, 'deltaMv', [])
    addParameter(iP, 'calibration_var', "Tcwout")
    
    parse(iP,varargin{:})
    
    gfc = iP.Results.gfc;
    logging = iP.Results.log;
    debug_pt = iP.Results.debug_pt;
    Tf = iP.Results.Tf;
    ann_deltaMv = iP.Results.ann_deltaMv;
    save_results = iP.Results.save_results;
    ptop_idx = iP.Results.ptop_idx;
    deltaMv = iP.Results.deltaMv;
    calibration_var = iP.Results.calibration_var;

    % Deactivate non necessary warnings
    warning('off', 'modelo_efecto1_LIQ:nonFeasibility')
    warning('off', 'waterPropertiesUtils:BPE');
    warning('off', 'waterPropertiesUtils:SpcHeat');
    
    % Load anns if not given as inputs
    if any([isempty(ann_deltaMv) ])
        anns = load("workspaces/anns_model_20220616.mat");
        if isempty(ann_deltaMv), ann_deltaMv = anns.net_deltaMv; end
        fprintf('Using ann from anns_model_20220616\n')
    end

    % Obtener deltaMv para variables de entrada
    if isempty(deltaMv)
        op_time = pt.timeOperated;
        deltaMv = evaluate_trained_ann([pt.Ms, pt.Ts_in, pt.Ts_out, pt.Mf, pt.Mcw, pt.Tcwin op_time], ann_deltaMv);
        fprintf('Using <strong>deltaMv=%.3f</strong> obtained from ANN\n', deltaMv)
%         deltaMv = debug_pt.processedData.deltaMv;
    else
        fprintf('Using deltaMv provided: %.2f\n', deltaMv)
    end
%     deltaMv=0.98;

%     [pt.Ms, pt.Ts_in, pt.Ts_out, pt.Mf, pt.Msw, pt.Tcwin];

    if isempty(Tf)
%         A = []; b = []; Aeq = []; beq = [];
%         opt = optimoptions(@fmincon,'display', 'off', ...
%                                     'Algorithm','sqp', ...
%                                     'OptimalityTolerance',1e-5, ...
%                                     'StepTolerance',1e-5, ...
%                                     'DiffMinChange', 0.5);                 
    
    %     ub0 = pt.Ts_out;
        success = false;
    %     lb = ub0-10; ub = ub0; lb_step = 0.1;
    %     
    %     while ub-lb > 0 && ~success
        idx=0; min_error = 999;
        for Tf=50:0.1:70
            idx=idx+1;
            try
    %             x0 = ub;
    %             
    %             Tf = fmincon(@(Tf) ajuste_Tf(pt.Tcwout, deltaMv, Tf, pt),...
    %                                x0,A,b,Aeq,beq,lb,ub,[],opt);
    %             success=true;
                error(idx) = ajuste_Tf(pt.(calibration_var), calibration_var, deltaMv, Tf, pt, debug_pt); %#ok<*AGROW> 
                if abs(error(idx))<min_error
                    min_error = abs(error(idx));
                    Tf_result = Tf;
                end
                success=true;
            catch ME 
    %             if strcmp(ME.identifier, "modelo_simulacion:NonValidTf")
    %                 rethrow(ME)
                    error(idx) = 9999;
                    warning("Tf=%.2f failed | %s: %s", Tf, ME.identifier, ME.message)
    %                 if ub-lb_step>lb
    %                     ub = ub-lb_step; 
    %                     warning('Upper bound %.2f for Tf produced an error, trying with a lower value (%.2f)', ub+lb_step, ub)
    %                 else 
    %                     ub = lb; 
    %                 end
                    
    %             else
    %                 rethrow(ME)
    %             end
            end
        end

        if success
            Tf = Tf_result; 
            fprintf('Using <strong>Tf=%.3f</strong>\n', Tf)
        end
    
    else
        success = true;
    end
    
    % Run simulation model
    if ~success
        ME = MException('modelo_simulacion_MED_PSA:no_solution_found', ...
                        ['Could not find a Tf that yields Tcwin ' ...
                        'for the given plant parameters']);
        throw(ME) 
    end

%     Tf = debug_pt.processedData.effects(1).Tb_in;
    [ef, ph, out] = modelo_base(Tf, deltaMv, pt, log=logging, gfc=false, debug_pt=debug_pt);

        % Log
        var="Tcwout"; T_ref = debug_pt.medidas.(var).value;
        if isempty(debug_pt)
            fprintf('<strong>Tf = %.3f</strong>  | Error = %6.2f [ºC]   |   %s_ref = %5.2f,  %s = %5.2f  [ºC]\n', ...
                    Tf, abs(T_ref - out.(var)), var, T_ref, var, out.(var));
        else
            fprintf(['<strong>Tf = %.3f</strong>    | Error %s = %6.2f [ºC]      |   %s_ref \t= %5.2f,  %s = %5.2f  [ºC]\n' ...
                                           '\t\t    | Error Mprod = %6.2f [kg/s] |   Mprod_ref\t = %5.2f,  Mprod = %5.2f  [kg/s]\n' ...
                                           '\t\t    | Error Tf = %6.2f [ºC]      |   Tf_ref\t = %5.2f,  Tf = %5.2f  [ºC]\n'], ...
                    Tf, var, T_ref - out.(var), var, T_ref, var, out.(var), ...
                    debug_pt.medidas.Mprod.value - out.Mprod, debug_pt.medidas.Mprod.value, out.Mprod, ...
                    debug_pt.medidas.Tf.value - Tf, debug_pt.medidas.Tf.value, Tf);
        end


    if gfc
        repModSim_comparison(ef, ph, out, debug_pt, save_results=save_results, ptop_idx=ptop_idx)
%         repModSim_results(ef, save_results=save_results, ptop_idx=ptop_idx)
    end
%     out.deltaMv = deltaMv;

end

function error = ajuste_Tf(T_ref, var, deltaMv, Tf, in, debug_pt)

    % Function that calculates the error between a given Tcwin and the one
    % obtained by running the simulation model for a given Tf (from fmincon)

    [~, ~, out] = modelo_base(Tf, deltaMv, in, log=false, gfc=false);

    % Log
    if isempty(debug_pt)
        fprintf('<strong>Tf = %.3f</strong>  | Error = %6.2f [ºC]   |   %s_ref = %5.2f,  %s = %5.2f  [ºC]\n', ...
                Tf, abs(T_ref - out.(var)), var, T_ref, var, out.(var));
    else
        fprintf(['<strong>Tf = %.3f</strong>    | Error %s = %6.2f [ºC]      |   %s_ref \t= %5.2f,  %s = %5.2f  [ºC]\n' ...
                                       '\t\t    | Error Mprod = %6.2f [kg/s] |   Mprod_ref\t = %5.2f,  Mprod = %5.2f  [kg/s]\n' ...
                                       '\t\t    | Error Tf = %6.2f [ºC]      |   Tf_ref\t = %5.2f,  Tf = %5.2f  [ºC]\n'], ...
                Tf, var, T_ref - out.(var), var, T_ref, var, out.(var), ...
                debug_pt.medidas.Mprod.value - out.Mprod, debug_pt.medidas.Mprod.value, out.Mprod, ...
                debug_pt.medidas.Tf.value - Tf, debug_pt.medidas.Tf.value, Tf);
    end

    error = sqrt((T_ref - out.(var))^2);

end


function [ef, ph, out] = modelo_base(Tf, deltaMv, in, varargin)
%----------------------------------------------------------------------%
%% -------------------Descripción de entradas-------------------------%%
%----------------------------------------------------------------------%
% deltaMv: Pérdida de vapor entre precalentador y siguiente efecto
%
% i_inicial: celda desde la que se comienza a ejecutar el modelo
% i_final: celda hasta la que se ejecuta el modelo
%
% EF_ANT, PH_ANT: si se ejecuta el modelo desde una celda distinta a la
% primera es necesario proveer los datos de la celda anterior a i_inicial
%
% in: Estructura que contiene las entradas necesarias para el modelo,
% se compone de:
%
% Nef: Número de efectos que componen la planta
% Ac: Área del condensador final [m2]
% Aph: Vector con el área de todos los precalentadores, incluyendo el
% condensador [m2]
% Aef: Vector con el área de todos los efectos [m2]
% Y: Apertura de las válvulas en los mixer
% ef_inst: Efectos instrumentados
% source: Fase del fluido caliente [liquid o vapour]
% MIXER: Presencia de mezcladores [true o false]
%
% Variables referentes al punto de operación contenidas en in:
% Md_ref: Flujo másico de referencia (i) en ef_inst(i) [kg/s]
% Tv_ref: Temperaturas de vapor en los efectos [ºC]
% Tph_ref: Temperaturas en los precalentadores de referencia [ºC]
% Ms: Flujo másico del fluido caliente [kg/s]
% Ts_in: Temperatura de entrada del fluido caliente [ºC]
% Ts_out: Temperatura de salida del fluido caliente [ºC]
% Ps: Presión del fluido caliente [bar]
% Mf: Flujo másico del agua de alimentación [kg/s]
% Tf: Temperatura del agua de alimentación a la entrada del 1er ef [ºC]
% Xf: Salinidad del agua de alimentación a la entrada del 1er ef [g/kg]
% Tcwin: Temperatura del agua a la entrada del condensador [ºC]
% Tcwout: Temperatura del agua a la salida del condensador [ºC]
% Msw: Flujo másico del agua de alimentación + refrigeración [kg/s]
% Tprod: Temperatura con la que el destilado abandona la planta [ºC]
%
%----------------------------------------------------------------------%
%% ---------------------Descripción de salidas------------------------%%
%----------------------------------------------------------------------%
% ef: Vector de objetos de tipo Efecto, contiene la información de
% interés de todos los efectos de la planta resultado de la ejecución
% del modelo. Para más información consultar la clase con "help Efecto"
%
% ph: Vector de objetos de tipo Preheater, contiene la información de
% interés de todos los precalentadores de la planta resultado de la
% ejecución del modelo. Para más información usar "help Preheater"
%
% out. Estructura que contiene las salidas de interés resultado de
% ejecutar el modelo. Éstas son:
% PR: Performance Ratio [KJ/s]
% STEC: Specific Thermal Energy Consumption [MJ/m3]
% Mprod: Flujo másico de destilado producido [kg/s]
% Tprod: Temperatura del destilado producido [ºC]
% Msw: Flujo másico del agua de alimentación + refrigeración [kg/s]
% Mcw: fujo másico del agua de refrigeración [kg/s]
% ---------------------------------------------------------------------

iP = inputParser;

check_grafica = @(x)validateattributes(x, {'logical'},{'scalar'});

addParameter(iP, 'gfc', false, check_grafica)
addParameter(iP, 'log', true, check_grafica)
addParameter(iP, 'debug_pt', [])

addParameter(iP, 'ann_Uef', [])
addParameter(iP, 'ann_Uph', [])
% addParameter(iP, 'ann_deltaMv', [])

parse(iP,varargin{:})

gfc      = iP.Results.gfc;
logging  = iP.Results.log;
debug_pt = iP.Results.debug_pt;

ann_Uef     = iP.Results.ann_Uef;
ann_Uph     = iP.Results.ann_Uph;
% ann_deltaMv = iP.Results.ann_deltaMv;

% Load anns if not given as inputs
if any([isempty(ann_Uef) isempty(ann_Uph)])
    anns = load("workspaces/anns_model_20220608.mat");
    if isempty(ann_Uef), ann_Uef = anns.net_Uef; end
    if isempty(ann_Uph), ann_Uph = anns.net_Uph; end
% if isempty(ann_deltaMv), ann_deltaMv = anns.net_deltaMv; end
end

% Figure setup if gfc true
if ~isempty(debug_pt) && gfc
    figure('Name','Preheaters','Color','#f8f9f9','WindowStyle','normal', ...
   'Units','normalized','Position',[0.5 0 0.5 1]);
    tl = tiledlayout("flow","TileSpacing","compact", "Padding","compact");
    xlabel(tl, 'Tph\_in')
    ylabel(tl, 'Preheater function output')
    title(tl, sprintf('Tf (%.4f) %+.2f', debug_pt.medidas.Tf.value, Tf-debug_pt.medidas.Tf.value))
end



%% -----------------Inicialización de variables-----------------------%%
% run inicializacion_cls.m
% opt_lqnonlin = optimoptions('lsqnonlin', 'Display','none');
warning('off', 'optim:fsolve:NonSquareSystem');
%     opt = optimoptions(@fmincon,'Display','none');

Nef = in.Nef;

if isscalar(Tf) == false        
    errID    = 'modelo_operacion:InputTf_notScalar';
    msg      = 'Tf no es un escalar.';
    Esepsion = MException(errID,msg);
    throw(Esepsion)
end
ef(Nef) = Efecto2;
ph(Nef) = Preheater2;

% Propiedades comunes a todos los efectos
k=1;
for i=1:length(ef)
    if i == in.ef_inst(k)
        ef(i).deltaMv = deltaMv(k);
        k = k+1;
    else
        ef(i).deltaMv = deltaMv(k);
    end
    ef(i).Aef = in.Aef(i);
    ef(i).op_time = in.timeOperated(i);
end

% Precalentadores 
for i=1:Nef
    ph(i).Mf = in.Mf;
    ph(i).Xf = in.Xf;
    ph(i).Aph = in.Aph(i);
end

% m=0; m2=0;

out = struct; out.Mprod = 0; out.Tprod = 0; out.Mcw_no_over_cooling = 0;

% Debug model inputs
if ~isempty(debug_pt)
    in_ref = debug_pt.medidas;
    disp(table([in_ref.Ms.value in.Ms], [in_ref.Ts_in.value in.Ts_in], ...
               [in_ref.Ts_out.value in.Ts_out], [in_ref.Ps.value in.Ps], ...
               [in_ref.Mf.value in.Mf], [in_ref.Tf.value Tf], ...
               [in_ref.Mcw.value in.Mcw], [in_ref.Tcwin.value in.Tcwin], ...
               [debug_pt.processedData.deltaMv deltaMv], ...
               VariableNames=["Ms", "Ts_in", "Ts_out", "Ps", "Mf", "Tf", "Mcw", "Tcwin", "deltaMv"]))
end
%%                           Cálculos                                    
final_effect=false; condenser=false;
for i=1:Nef
    
    % EFFECT
        % Set inputs
    
        % Particular
        if i == 1
            ef(i).Ms = in.Ms;
            ef(i).Ts_in = in.Ts_in;
            ef(i).Tb_in = Tf;
            ef(i).Xb_in = in.Xf;
            ef(i).Mb_in = in.Mf;
            first_effect = true;
    
            if in.source == "liquid"
                ef(i).Ts_out = in.Ts_out;
                ef(i).Ps = in.Ps;
                ef(i).Cp_s = cpW(in.Ts_in, in.Ps);
    
            elseif in.source == "steam"
    
            else
                Esepsion = MException('modelo_simulacion_new:unkownInput', ...
                           'Unrecognized input %s for source type', in.source);
                throw(Esepsion)
            end
            
    
            
        else
            first_effect = false;
            if i==Nef, final_effect=true; end

            ef(i).Mv_in  = ef(i).deltaMv*ph(i-1).Mv_out;
            ef(i).Tv_in  = pipe_losses(ef(i).Mv_in, ph(i-1).Tv);

            ef(i).Mdelta = (1-ef(i).deltaMv)*ph(i-1).Mv_out;
    
            ef(i).Mvh    = ph(i-1).Mvh;
    
            ef(i).Mb_in  = ef(i-1).Mb_out;
            ef(i).Tb_in  = ef(i-1).Tb_out;
            ef(i).Xb_in  = ef(i-1).Xb_out;
    
            % Particular inputs due to distribution line
            m  = ef(i-1).m;
            m2 = ef(i-1).m2;
            
            if i == 3*m+2 % [5,8,11,14] 
                % No entra destilado del efecto anterior: Md_in = 0, Td_in = 0
                % No entra destilado del mixer: Mmix_in = 0, Tmix_in = 0
                ef(i).Md_in = 0;
                ef(i).Td_in = 0;
                ef(i).Mda = 0;
                if final_effect
                    ef(i).Mdb     = ef(i-1).Mmix_out;
                    ef(i).Tmix_in = ef(i-1).Tmix_out;
                else
                    ef(i).Mdb = 0;
                    ef(i).Tmix_in = 0;
                end
                
                m=m+1;
    
            elseif i == 3*m2+4 % [7,10,13]
                % Entra destilado del mixer
                % Entra destilado del efecto anterior
                ef(i).Y = in.Y(m2);
                if m2 == 1 % [7]  
                    ef(i).Mda = ef(i-3).Md_out*ef(i).Y;
                    ef(i).Mdb = ef(i-3).Md_out*(1-ef(i).Y);
                    ef(i).Tmix_in = ef(i-1).Td_out;
                else       % [10,13]      
                    ef(i).Mda = ef(i-3).Mmix_out*ef(i).Y;
                    ef(i).Mdb = ef(i-3).Mmix_out*(1-ef(i).Y);
                    ef(i).Tmix_in = ef(i-3).Tmix_out;
                end
                ef(i).Md_in = ef(i-1).Md_out;
                ef(i).Td_in = ef(i-1).Td_out;
                m2=m2+1;
                
            else    % [2-4, 6, 9, 12]
                % No entra destilado del mixer: Mmix_in = 0, Tmix_in = 0
                % Entra destilado del efecto anterior
                ef(i).Mda = 0;
                ef(i).Mdb = 0;
                ef(i).Tmix_in = 0;
                ef(i).Md_in = ef(i-1).Md_out;
                ef(i).Td_in = ef(i-1).Td_out;
            end
    
        end
        
        if ~first_effect
            % Update counters
            ef(i).m = m;
            ef(i).m2 = m2;

            % Total condensate input
            ef(i) = ef(i).calculoScpHeat();
            % Balance de masa:
            ef(i).Mdest_in = ef(i).Mvh + ef(i).Mda + ef(i).Md_in + ef(i).Mdelta;
            % Balance de energía:
            ef(i).Tdest_in = ( ef(i).Mvh * ef(i).Tv_in * ef(i).Cp_ph +...
                               ef(i).Mdelta * ef(i).Tv_in * ef(i).Cp_delta +...
                               ef(i).Md_in * ef(i).Td_in * ef(i).Cp_d +...
                               ef(i).Mda * ef(i).Tmix_in * ef(i).Cp_mix ) /...
                             ( ef(i).Mdest_in * ef(i).Cp_ph );
        end

        % Debug inputs
        if ~isempty(debug_pt)
            fprintf('<strong>______________________________________________________________________________________________________________________________________________________</strong>\n')
            fprintf('                                                                              <strong> CELL %d  </strong>\n', i)
            fprintf('<strong>______________________________________________________________________________________________________________________________________________________</strong>\n\n')
            ef_ref = debug_pt.processedData.effects(i);
            disp(table([ef_ref.Ms ef(i).Ms], [ef_ref.Ts_in ef(i).Ts_in], ...
                       [ef_ref.Ts_out ef(i).Ts_out], [ef_ref.Mdest_in ef(i).Mdest_in], ...
                       [ef_ref.Tdest_in ef(i).Tdest_in], [ef_ref.Mv_in ef(i).Mv_in], ...
                       [ef_ref.Tv_in ef(i).Tv_in], [ef_ref.Mb_in    ef(i).Mb_in], ...
                       [ef_ref.Tb_in ef(i).Tb_in], [ef_ref.Xb_in ef(i).Xb_in], ...
                       VariableNames=[sprintf("Ms(%d)", i), sprintf("Ts_in(%d)",i), ...
                                      sprintf("Ts_out(%d)", i), sprintf("Mdest_in(%d)",i), ...
                                      sprintf("Tdest_in(%d)", i), sprintf("Mv_in(%d)",i), ...
                                      sprintf("Tv_in(%d)",i),     sprintf("Mb_in(%d)",i), ...
                                      sprintf("Tb_in(%d)",i),     sprintf("Xb_in(%d)",i)]))
        end

        % Calculations
        if ~isempty(debug_pt), ef_ref = debug_pt.processedData.effects(i); else, ef_ref = []; end

        ef(i) = effect_simulation_block(ef(i), ann_Uef, source="liquid", first_effect=first_effect, final_effect=final_effect, ef_ref=ef_ref);
        
        % Debug outputs
        if ~isempty(debug_pt)
            ef_ref = debug_pt.processedData.effects(i);

            if i==1
                disp(table([ef_ref.Tb_out ef(i).Tb_out abs(ef(i).Tb_out-ef_ref.Tb_out)/ef_ref.Tb_out*100], ...
                           [ef_ref.Mgb ef(i).Mgb abs(ef(i).Mgb-ef_ref.Mgb)/ef_ref.Mgb*100], ...
                           [ef_ref.Mb_out ef(i).Mb_out abs(ef(i).Mb_out-ef_ref.Mb_out)/ef_ref.Mb_out*100], ...
                           [ef_ref.Xb_out ef(i).Xb_out abs(ef(i).Xb_out-ef_ref.Xb_out)/ef_ref.Xb_out*100], ...
                           [ef_ref.Tv_out ef(i).Tv_out abs(ef(i).Tv_out-ef_ref.Tv_out)/ef_ref.Tv_out*100], ...
                           [debug_pt.medidas.Tf.value Tf], ...
                           [ef_ref.Uef    ef(i).Uef], ...
                           VariableNames=["Tb(1)", "Mgb(1)", "Mb(1)", "Xb(1)", "Tv(1)", "Tf", "Uef"]))
            else
                disp(table([ef_ref.Tb_out ef(i).Tb_out abs(ef(i).Tb_out-ef_ref.Tb_out)/ef_ref.Tb_out*100], ...
                           [ef_ref.Mgb ef(i).Mgb abs(ef(i).Mgb-ef_ref.Mgb)/ef_ref.Mgb*100], ...
                           [ef_ref.Mb_out ef(i).Mb_out abs(ef(i).Mb_out-ef_ref.Mb_out)/ef_ref.Mb_out*100], ...
                           [ef_ref.Xb_out ef(i).Xb_out abs(ef(i).Xb_out-ef_ref.Xb_out)/ef_ref.Xb_out*100], ...
                           [ef_ref.Tv_out ef(i).Tv_out abs(ef(i).Tv_out-ef_ref.Tv_out)/ef_ref.Tv_out*100], ...
                           [ef_ref.Md_out ef(i).Md_out abs(ef(i).Md_out-ef_ref.Md_out)/ef_ref.Md_out*100], ...
                           [ef_ref.Mgf ef(i).Mgf abs(ef(i).Mgf-ef_ref.Mgf)/ef_ref.Mgf*100], ...
                           [ef_ref.Uef    ef(i).Uef], ...
                           VariableNames=[sprintf("Tb_out(%d)", i), sprintf("Mgb(%d)",i), ...
                                          sprintf("Mb_out(%d)", i), sprintf("Xb_out(%d)",i), ...
                                          sprintf("Tv_out(%d)", i), sprintf("Md_out(%d)",i), ...
                                          sprintf("Mgf(%d)",i),     sprintf("Uef(%d)",i)]))
            end
        end


    % Set inputs
 
    % Common
    ph(i).Mv_in = ef(i).Mv_out;
    ph(i).Tv = demister(ef(i).Mv_out, ef(i).Tv_out);
    ph(i).Cp = SW_SpcHeat(ph(i).Tph_out, ph(i).Xf)/1000;

    % Particular
    if ~first_effect
        ph(i).Tph_out = ph(i-1).Tph_in;
    else
        ph(i).Tph_out = Tf;
    end

    % PREHEATER
    if i<Nef

        
    % CONDENSER
    else
        condenser=true;

        ph(i).Mcw = in.Mcw;
%         if ~isfield(in, "Mcw")
%             ph(i).Mcw = in.Msw-4; % Los 4 m³/h que van al sistema de vacío
%         elseif ph(i).Mcw == 0
%             ph(i).Mcw = in.Msw-4;
%         end
%             ef.Mmix_out = ef.Mdb + ef.Md_out;
%             ef.Tmix_out = ( ef.Mdb * ef.Tmix_in * ef.Cp_mix + ef.Md_out ...
%                             * ef.Td_out * cpSatLiqTW(ef.Td_out) ) ...
%                           / (ef.Mmix_out * ef.Cp_d);
        ph(i).Mmix_in = ef(i).Mmix_out;
        ph(i).Cp_mix  = ef(i).Cp_mix;
        ph(i).Tmix_in = ef(i).Tmix_out;
    end

    % Check inputs
    if ph(i).Tv < ph(i).Tph_out
        ME = MException('modelo_simulacion_new:invalidTvc', ...
             ['Vapor temperature in preheater lower than exit ' ...
             'temperature: Tv (%.2f) < Tph_out (%.2f)'], ph(i).Tv, ph(i).Tph_out);
        throw(ME)
    end

    % Calculations
    if ~isempty(debug_pt), ph_ref = debug_pt.processedData.preheaters(i); else, ph_ref = []; end
    ph(i) = preheater_simulation_block(ph(i), ann_Uph, condenser=condenser, ph_ref=ph_ref);

    % Debug
    if ~isempty(debug_pt)
        ph_ref = debug_pt.processedData.preheaters(i);

        disp(table([ph_ref.Tv ph(i).Tv], [ph_ref.Tph_in ph(i).Tph_in], [ph_ref.Tph_out ph(i).Tph_out], ...
                   [ph_ref.Mvh ph(i).Mvh], [ph_ref.Mv_out ph(i).Mv_out], [ph_ref.Uph ph(i).Uph], ...
                   VariableNames=[sprintf("Tv(%d)", i),      sprintf("Tph_in(%d)", i), ...
                                  sprintf("Tph_out(%d)", i), sprintf("Mvh(%d)", i), ...
                                  sprintf("Mv_out(%d)", i),  sprintf("Uph(%d)", i)]))
        
%         Tin = ph(i).Tph_out-10:0.1:ph(i).Tph_out;
%         for idx=1:length(Tin)
%             y(idx, 1) = precalentador(Tin(idx), ph(i).Uph, ph(i).Aph, ph(i).Tv, ph(i).Tph_out, ph(i).Mf, ph(i).Cp); %#ok<*AGROW> 
%             y(idx, 2) = precalentador(Tin(idx), ph(i).Uph+0.5, ph(i).Aph, ph(i).Tv, ph(i).Tph_out, ph(i).Mf, ph(i).Cp);
%             y(idx, 3) = precalentador(Tin(idx), ph(i).Uph-0.5, ph(i).Aph, ph(i).Tv, ph(i).Tph_out, ph(i).Mf, ph(i).Cp);
%         end
%         ax=nexttile;
%         plot(ax, Tin, y); hold on; yline(0, 'HandleVisibility','off');
%         scatter(ax, ph_ref.Tph_in, 0, 35, [0.4660 0.6740 0.1880], "filled",LineWidth=1.5)
%         scatter(ax, tphin, precalentador(tphin, ph(i).Uph, ph(i).Aph, ph(i).Tv, ph(i).Tph_out, ph(i).Mf, ph(i).Cp), 35, [0 0.4470 0.7410], Marker="x", LineWidth=1.5)
%         scatter(ax, tphin_aux, precalentador(tphin_aux, ph(i).Uph, ph(i).Aph, ph(i).Tv, ph(i).Tph_out, ph(i).Mf, ph(i).Cp), 35, [0 0.4470 0.7410], Marker="diamond", LineWidth=1.5)
        %                     lg = legend(ax, sprintf('U=%.2f', Uph(i)), 'U+0.5', 'U-0.5', 'Ref', Location='best'); %#ok<*NASGU> 
        %                     lg.Orientation = 'horizontal';
    
    end

    % RESULTS
    if i==Nef
        Mprod = ph(end).Mvh;
        Tprod = ph(end).Tvh;

        % Performance parameters
        % PR
        if strcmp(in.source, 'liquid')
            PR = Mprod * 2326 / (in.Ms * (in.Ts_in-in.Ts_out) * ef(1).Cp_s); % [KJ/s]
        else
            PR = Mprod * 2326 / (in.Ms * (enthalpySatVapTW(in.Ts_in)-enthalpySatLiqTW(in.Ts_in))); % [KJ/s]
        end

        % STEC
        STEC = 2326/PR * densSatLiqTW(Tprod) * 1e-3; % [MJ/m3]
        
        out.Mprod = ph(end).Mvh; 
        out.Tprod = ph(end).Tvh; 
        out.Mcw_no_over_cooling = ph(end).Mcw_no_over_cooling;
        out.STEC  = STEC; out.PR = PR; 
        out.Tcwin  = ph(end).Tph_in; 
        out.Tcwout = ph(end).Tph_out;
        out.Tf = Tf;
        out.delaMv = deltaMv;


        % Display results
        % Display table with comparison with debug pt
%         if ~isempty(debug_pt), summary_output_comparison_table(Nef, ef, ph, debug_pt), end 
        % Display table with results
%         if logging, summary_output_table(in, ef, ph, out); end
        % Results visualization
%         if gfc
%             ef_input_energy_breakdown(ef, in)
%             results_visualization
%         end
    end

end
    
end

function summary_output_comparison_table(Nef, ef, ph, debug_pt)
    for i=1:Nef
        % Effect comparison
        ef_ref = debug_pt.processedData.effects(i);
        disp(table([ef_ref.Tb_out ef(i).Tb_out], [ef_ref.Mgb ef(i).Mgb], ...
                   [ef_ref.Mb_out ef(i).Mb_out], [ef_ref.Xb_out ef(i).Xb_out], ...
                   [ef_ref.Tv_out ef(i).Tv_out], [ef_ref.Md_out ef(i).Md_out], ...
                   [ef_ref.Mgf ef(i).Mgf], ...
                   VariableNames=[sprintf("Tb_out(%d)", i), sprintf("Mgb(%d)",i), ...
                                  sprintf("Mb_out(%d)", i), sprintf("Xb_out(%d)",i), ...
                                  sprintf("Tv_out(%d)", i), sprintf("Md_out(%d)",i), ...
                                  sprintf("Mgf(%d)",i)]))
        % Preheater comparison
        ph_ref = debug_pt.processedData.preheaters(i);
        disp(table([ph_ref.Tph_in ph(i).Tph_in], [ph_ref.Tph_out ph(i).Tph_out], ...
                   [ph_ref.Mvh ph(i).Mvh], [ph_ref.Mv_out ph(i).Mv_out], ...
                   VariableNames=[sprintf("Tph_in(%d)", i), sprintf("Tph_out(%d)", i), sprintf("Mvh(%d)", i), sprintf("Mv_out(%d)", i)]))
    end
end

function summary_output_table(in, ef, ph, out)
    % Tabla resumen de resultados
    fprintf("\n____________________________________________________\tMODELO ESTÁTICO MED PSA\t________________________________________________________ \n");
    fprintf("                    Condiciones de partida: Ms = %.2f [kg/hr] | Ts_in = %.1f [ºC] | Mf = %.2f [kg/hr] | Tf = %.2f [ºC]\n",...
        in.Ms*3600, in.Ts_in, in.Mf*3600, in.Tf);
    fprintf("___||________________________________________________EFECTO________________________________ _______________||______PRECALENTADOR_________\n")
    fprintf("   ||             M [kg/hr]                    |                   T [ºC]                  |    X   [g/kg] ||  M [kg/hr]   |    T [ºC] \n");
    fprintf("---||------------------------------------------------------------------------------------------------------||---------------------------\n")
    fprintf("   ||   Mgb    Mgf      Mb       Md       Mv   |  Tv     Tb     Td      Tv'    Tb'    Td'  |   Xb      Xb' ||  Mvh    Mv   |  Tin   Tout\n");
    fprintf("---||------------------------------------------------------------------------------------------------------||---------------------------\n")
    i=1;fprintf("%2d || %5.2f  %5.2f  %5.2f  %7.2f  %5.2f | %5.2f  %5.2f  %5.2f  %5.2f  %5.2f  %5.2f  | %5.2f   %5.2f || %5.2f %5.2f | %5.2f  %5.2f \n",...
        1, ef(i).Mgb*3600, ef(i).Mgf*3600, ef(i).Mb_out*3600, ef(i).Md_out*3600, ef(i).Mv_out*3600,...
        ef(i).Tv_out, ef(i).Tb_out, ef(i).Td_out, 0,0,0, ef(i).Xb_out, ef(i).Xbb,...
        ph(i).Mvh*3600, ph(i).Mv_out*3600, ph(i).Tph_in, in.Tf)
    for i=2:in.Nef-1
    fprintf("%2d || %5.2f  %5.2f  %5.2f  %7.2f  %5.2f | %5.2f  %5.2f  %5.2f  %5.2f  %5.2f  %5.2f  | %5.2f   %5.2f || %5.2f %5.2f | %5.2f  %5.2f \n",...
        i, ef(i).Mgb*3600, ef(i).Mgf*3600, ef(i).Mb_out*3600, ef(i).Md_out*3600, ef(i).Mb_out*3600,...
        ef(i).Tv_out,ef(i).Tb_out,ef(i).Td_out,0,0,0, ef(i).Xb_out, ef(i).Xbb,...
        ph(i).Mvh*3600, ph(i).Mv*3600, ph(i).Tph_in, ph(i).Tph_out)
    end
    i=in.Nef;fprintf("%2d || %5.2f  %5.2f  %5.2f  %7.2f  %5.2f | %5.2f  %5.2f  %5.2f  %5.2f  %5.2f  %5.2f  | %5.2f   %5.2f || %5.2f %4.1f | %5.2f  %5.2f \n",...
        in.Nef, ef(i).Mgb*3600, ef(i).Mgf*3600, ef(i).Mb_out*3600, ef(i).Md_out*3600, ef(i).Mb_out*3600,...
        ef(i).Tv_out,ef(i).Tb_out,ef(i).Td_out,0,0,0, ef(i).Xb_out, ef(i).Xbb,...
        out.Mprod*3600, 0, ph(i).Tph_in, ph(i).Tph_out)
    fprintf("________________________________________________________________________________________________________________________________________\n")
    fprintf("                    Condiciones de salida: Mprod = %.2f [kg/hr] | Tprod = %.1f [ºC] | Msw = %.2f [kg/hr] | Mcw = %.2f [kg/hr]\n",...
        out.Mprod*3600, out.Tprod, out.Msw*3600, out.Mcw*3600);
    fprintf("________________________________________________________________________________________________________________________________________\n")
    fprintf("                                                               STEC = %.2f [MJ/m3],  PR = %.2f \n", out.STEC, out.PR)
    fprintf("________________________________________________________________________________________________________________________________________\n")
end

function ef_input_energy_breakdown(efs, in)
    figure
    plot([efs.aporte_v_ant])
    hold on
    plot([efs.aporte_ef_ant])
    plot([efs.aporte_ph_ant])
    plot([efs.aporte_b])
    plot([efs.aporte_mix])
    plot([efs.aporte_delta])
    label = [num2cell(100*in.Y(1))+" %" num2cell(100*in.Y(2))+" %" num2cell(100*in.Y(3))+" %"];
    text([7,10,13], [efs([7,10,13]).aporte_mix],label,'VerticalAlignment','bottom','HorizontalAlignment','center')
    ax = gca;

    title('Aporte generación de vapor por evaporación')
    xlabel("Efectos")
    ylabel("Energía aportada [KJ]")
    grid on
    legend('Vapor anterior','Dest. efs. anterior','Dest. phs. anterior','Salmuera',"Mixer",'DeltaMv','location','best')
    ax.YMinorGrid = 'on';
end

function results_visualization(in, ef, ph)
    figure('Name','Preheaters','Color','#f8f9f9','WindowStyle','normal', ...
           'Units','normalized','Position',[0.5 0 0.5 1]);
    
    tl = tiledlayout(2,4,"TileSpacing","compact", "Padding","compact");
    
    ax = nexttile; hold(ax, "on");
    plot([ef.Tv_out])
    % xline(Nef,"--")
    % yline(Tvc,"--")
    % plot(Nef, Tvc, "ro")
    plot(ax, in.Tv_des, 'x')
    title(ax, "Evolución Tv")
    ax.XGrid = 'on';ax.XTick=[1:in.Nef];ax.YMinorGrid = 'on'; %#ok<*NBRAK> 
    
    ax = nexttile; hold(ax, "on");
    plot(ax, [ph.Tph_in])
    plot(ax, in.Tph_des, 'x')
    title(ax, "Evolución Tph")
    ax.XGrid = 'on'; ax.XTick=[1:in.Nef]; ax.YMinorGrid = 'on';
    
    ax = nexttile; hold(ax, "on");
    plot([ph(1:end-1).Mvh]*3600)
    plot([ef(2:end).Mgf]*3600)
    title(ax, "Evolución Mvh, Mgf")
    legend(ax, 'Mvh', 'Mgf','location','best')
    ax.XGrid = 'on'; ax.XTick=[1:in.Nef]; ax.YMinorGrid = 'on';
    
    ax = nexttile; hold(ax, "on");
    plot(ax, [ef.Xb_out])
    title(ax, "Evolución Xb")
    ax.XGrid = 'on'; ax.XTick=[1:in.Nef]; ax.YMinorGrid = 'on';
    
    ax = nexttile; hold(ax, "on");
    plot(ax, [ef.Mgb]*3600)
    plot(ax, [ph(1:end-1).Mv]*3600)
    plot(ax, ([ef.Mgb]+[ef.Mgf])*3600)
%     plot(in.Mgen_ref, 'x')
    title(ax, "Evolución Mgb, Mv")
    legend(ax, 'Mgb', 'Mv','Mgb+Mgf','ref','location','best')
    ax.XGrid = 'on'; ax.XTick=[1:in.Nef]; ax.YMinorGrid = 'on';
    
    ax = nexttile; hold(ax, "on");
    plot(ax, [ef.Mb_out]*3600)
%     plot(in.Mb_ref,'x')
    title(ax, "Evolución Mb")
    ax.XGrid = 'on'; ax.XTick=[1:in.Nef]; ax.YMinorGrid = 'on';
    
%     subplot(2,4,7)
%     
%     hold on
%     title("Evolución Mv")
%     ax = gca;ax.XGrid = 'on';ax.XTick=[1:in.Nef];ax.YMinorGrid = 'on';
    
    ax = nexttile(tl, 6, [2 1]); hold(ax, "on");
    mmixout = [ef(1:in.Nef-1).Mmix_out]; mmt = mmixout ~= 0;
    mdout = [ef(1:in.Nef-1).Md_out]; mdout(mmt)=0;
    dout = [mdout + mmixout [ph(in.Nef).Mvh] ];
%     error = abs(in.Md_ref - dout([4,7,14]))*3600;
%     label = num2cell(round(error*100)/100);
    
    bar(ax, dout*3600,'FaceAlpha',0.3,'EdgeAlpha',0.2,'BarWidth',0.5,'FaceColor','#EDB120')
    bar(ax, [ef.Md_out]*3600,'FaceAlpha',0.7,'EdgeAlpha',0.2,'BarWidth',0.5)
%     text([4,7,14], in.Md_ref*3600,label,'VerticalAlignment','bottom','HorizontalAlignment','right')
    plot(ax, [4,7,14], in.Md_ref*3600,'x','MarkerSize',15,'Color','#D95319');
    plot(ax, dout*3600,'LineWidth',2,'HandleVisibility','off','Color','#A2142F')
    
    title(ax, "Evolución destilado generado")
    legend(ax, 'Dest. ef','Dest. mix','Referencias','location','northwest')
    ax.XGrid = 'on';ax.XTick=[1:in.Nef];ax.YMinorGrid = 'on';
end
