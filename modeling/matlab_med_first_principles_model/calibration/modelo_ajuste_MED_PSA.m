function [ef, ph, out] = modelo_ajuste_MED_PSA(deltaMv,i_inicial, i_final,EF_ANT,PH_ANT, in, varargin)
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
    check = @(x)validateattributes(x, {'logical'},{'scalar'});
    
    addParameter(iP, 'gfc', false, check)
    addParameter(iP, 'log', true, check)
    addParameter(iP, 'debug', false, check)
    
    parse(iP,varargin{:})
    
    gfc = iP.Results.gfc;
    logging = iP.Results.log;
    debug = iP.Results.debug;


    %----------------------------------------------------------------------%
    %% -----------------Inicialización de variables-----------------------%%
    %----------------------------------------------------------------------%
    
    % run inicializacion_cls.m
%     clc;
    options_fsolve = optimoptions('fsolve','Display','none', 'Algorithm','trust-region');
    options_fmincon = optimoptions(@fmincon,'Display','none', 'Algorithm','sqp');
    warning('off', 'optim:fsolve:NonSquareSystem');
    warning('off', 'waterPropertiesUtils:BPE');
    warning('off', 'waterPropertiesUtils:SpcHeat');
    warning('off', 'modelo:efecto1_LIQ')
    warning('off', 'modelo_efecto1_LIQ:nonFeasibility')
    
    Ms = in.Ms; Ts_in = in.Ts_in; Mf = in.Mf; Tf = in.Tf; Xf = in.Xf; Aef = in.Aef; %#ok<NASGU>
    Aph = in.Aph; Nef = in.Nef; Tv_ref = in.Tv_ref; Tph_ref = in.Tph_ref;

    if ~all(diff(Tph_ref) <= -0.5)
       error('modelo_ajuste_MED_PSA:invalid_inputs', "Temperature differences in preheaters must be strictly decreasing with a minimum step of 0.5 °C. Found: %s ºC\n", mat2str(diff(Tph_ref), 3));
    end
    
    % Al ser una función cuando se inicia desde un efecto distinto del primero
    % se requiere dar como entrada el efecto anterior
    if i_inicial > 1
        ef(i_inicial-1) = EF_ANT;
        ef(i_inicial:i_final) = Efecto;

        ph(i_inicial-1) = PH_ANT;
        ph(i_inicial:i_final) = Preheater;
    else
        ef(i_inicial:i_final) = Efecto;
        ph(i_inicial:i_final) = Preheater;
    end

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
        ef(i).source = in.source;
    end

    % Propiedades comunes a todos los precalentadores
    for i=1:length(ph)
        ph(i).Aph = in.Aph(i);
        ph(i).Mf = Mf;
        ph(i).Xf = Xf;
    end

    out = struct; out.Mprod = 0; out.Tprod = 0;
    out.Msw = 0; out.Mcw = 0;

    if gfc
        figure('Name','Preheaters','Color','#f8f9f9','WindowStyle','normal', ...
            'Units','normalized','Position',[0.5 0 0.5 1]);
        t = tiledlayout(Nef,3,"TileSpacing","none", "Padding","compact");
        title(t,'Preheaters evolution');
    end


%-------------------------------------------------------------------------%
%%                           Cálculos                                    %%
%-------------------------------------------------------------------------%
    
    for i=i_inicial:i_final
% -------------------------------------------------------------------------
%%                             EFECTO 1                                  %%
%--------------------------------------------------------------------------
        if i==1
            % Inicialización de clase con entradas
            ef(i).Ts_in = in.Ts_in;
            ef(i).Ms = in.Ms;
            ef(i).Tb_in = in.Tf;
            ef(i).Xb_in = in.Xf;
            ef(i).Mb_in = in.Mf;
            
            if strcmp(in.source, 'liquid')
                ef(i).Ps = in.Ps;
                ef(i).Ts_out = in.Ts_out;
                ef(i).Cp_s = cpW(in.Ts_in, in.Ps);
            end
                        
            % Cálculos del efecto
            % x0 = [1 2 3 4 5];
            x0 = 0.01:0.01:1.5; x0 = [x0 1.6:5];
            A = []; b = []; Aeq = []; beq = []; 
            lb = 0.1; ub = 5;
            fun = @(U)calculoUef(U, Tv_ref(i), i, ef(i));

            for idx=1:length(x0)
                Uff(idx) = fmincon(fun,x0(idx),A,b,Aeq,beq,lb,ub,[],options_fmincon);
            end

            % Filter out repetead solutions
            idx = [true abs(diff(Uff))>0.01]; if ~any(idx), Uff=Uff(1); else, Uff=Uff(idx); end

            % And evaluate the output for each solution
            % For high temperature tests (where there is a high T diff 
            % between Tsin and Tb or Tv), it's better to have initial guess
            % based on Tf+detaT not Tsin-deltaT
            if ef(i).Tb_in+4 < Ts_in
                Tv_out0 = ef(i).Tb_in+4;
                Tb_out0 = ef(i).Tb_in+4;
            else
                Tv_out0 = ef(i).Ts_in-5;
                Tb_out0 = ef(i).Ts_in-5;
            end
%             disp(Tv_out0)
            x0 = [Tb_out0,ef(i).Mb_in*0.1,ef(i).Mb_in*0.9,ef(i).Xb_in*1.1,Tv_out0];

            fval = []; a_temp = [];
            for idx=1:length(Uff)
            if strcmp(in.source, 'liquid')
                a_temp(idx,:) = fsolve(@(x) efecto_1_LIQ(x, Uff(idx), ef(i).Aef,...
                    ef(i).Ms, ef(i).Ts_in,ef(i).Ts_out, ef(i).Tb_in,...
                    ef(i).Mb_in, ef(i).Xb_in, ef(i).Cp_s),x0,options_fsolve);

            elseif strcmp(in.source, 'steam')
                a_temp(idx,:) = fsolve(@(x) efecto_1(x, Uff(idx), ef(i).Aef, ef(i).Ms,...
                    ef(i).Ts_in, ef(i).Tb_in, ef(i).Mb_in, ef(i).Xb_in),x0,options_fsolve);
            end
%                 [Uff(idx), a_temp(5) Tv_ref(i)]
                fval(idx)       = abs( Tv_ref(i) - a_temp(idx,5) );
            end
            
            % The correct solution is the one that returns the minimum value
            ef(i).Uef = Uff(fval==min(fval));
            a =      a_temp(fval==min(fval),:);

            [ef(i).Tb_out, ef(i).Mgb, ef(i).Mb_out, ef(i).Xb_out, ef(i).Tv_out] = deal(a(1),a(2),a(3),a(4),a(5));
            ef(i).bpe = SW_BPE(ef(i).Tb_out,ef(i).Xb_out);

            error_ = abs(ef(i).Tv_out - Tv_ref(i));
            if error_ > 0.1
                error('calibrationModel:effectErrorThreshold', ...
                    ['The difference between the reference effect tempererature ' ...
                    'and the calculated one is too high (%.2f) for cell %d'], error_,i)
            end
           
            % Obtain energy input for each source
            ef(i) = ef(i).estimacionAportes2;

            %  -------------------------- log -----------------------------
            if logging
            fprintf('\n_________________________________________ CELDA %2d ___________________________________________\n',i)
            fprintf('\nEF | Error = %4.2f [ºC]  |  Tref=%4.2f,    T=%4.2f [ºC]  |  U=%5.2f [kW/m2ºC]\n', ...
                error_, Tv_ref(i), ef(i).Tv_out, ef(i).Uef);            
            end
            % -------------------------------------------------------------
            
% -------------------------------------------------------------------------
%%                           PRECALENTADOR 1                             %%
%--------------------------------------------------------------------------
            % Inicialización de entradas
            ph(i) = ph(i).setInputs(ef(i));
            ph(i).Tph_out = Tf;
            ph(i).Cp = SW_SpcHeat(ph(i).Tph_out, ph(i).Xf)/1000;

            % Check
            if ph(i).Tv < ph(i).Tph_out
                error('calibrationModel:invertedHE',['Error in preheater' ...
                    '%d, steam temperature in preheater (Tv) is smaller' ...
                    ' than feed water ouput temperature (Tph_out): ' ...
                    '%.2f < %.2f'], i, ph(i).Tv, ph(i).Tph_out)
            end
            
            % Cálculos del precalentador
            % (el código es igual que para un precalentador cualquiera, 
            % igual sería interesante no tener que repetirlo dos veces)
            
            x0 = 0.1:10;
            A = []; b = []; Aeq = []; beq = []; 
            lb = 0.1; ub = 20;
            fun = @(U)calculoUph_cls(U,Tph_ref(i+1),i,Nef,ph(i));

            Uff = [];
            for idx=1:length(x0)
                Uff(idx) = fmincon(fun,x0(idx),A,b,Aeq,beq,lb,ub,[],options_fmincon);                
            end             
            
            % Filter out repetead solutions
            idx = [true abs(diff(Uff))>0.01]; if ~any(idx), Uff=Uff(1); else, Uff=Uff(idx); end

            % And evaluate the output value
            fval = [];
            for idx=1:length(Uff)
                fval(idx) = fun(Uff(idx));
            end
            % The correct solution is the one that returns the minimum value
            ph(i).Uph = Uff(fval==min(fval));
            
            ph(i).Tph_in = fsolve(@(Tin) ...
            precalentador(Tin, ph(i).Uph, ph(i).Aph, ph(i).Tv, ...
            ph(i).Tph_out, ph(i).Mf, ph(i).Cp), ph(i).Tph_out-5, options_fsolve);

            error_ = abs(ph(i).Tph_in - Tph_ref(i+1));
            % ------------------------------------- log ----------------------------------------------------------------- 
            if logging
            fprintf('PH | Error = %4.2f [ºC]  |   Tin=%4.2f, Tout=%4.2f [ºC]  |  U=%5.2f [kW/m2ºC] | Tv=%5.2f [ºC]\n', ...
                error_, ph(i).Tph_in, ph(i).Tph_out, ph(i).Uph, ph(i).Tv);
            end
            % -----------------------------------------------------------------------------------------------------------

            if error_ > 0.1
                if debug, evaluacionPh(in, i, ph), end

                error('calibrationModel:preheaterErrorThreshold', ...
                      ['The difference between the reference preheater tempererature ' ...
                      'and the calculated one is too high (%.2f ºC) for cell %d'], error_,i)
            end

            ph(i).Mvh = ( ph(i).Mf*ph(i).Cp*(ph(i).Tph_out-ph(i).Tph_in) ) / ...
                        ( enthalpySatVapTW(ph(i).Tv)-enthalpySatLiqTW(ph(i).Tv) );
            ph(i).Mv = ph(i).Mgb - ph(i).Mvh; % sería mejor que se llamase Mv_out

            if gfc, preheaters_evolution(t,i,ph(i),Tph_ref(i+1),ef(i), ef, ph); end
        
% -------------------------------------------------------------------------
%%                             EFECTO i                                  %%
%--------------------------------------------------------------------------
        elseif i>1

            % Inicialización de entradas común para todos los efectos
            ef(i) = ef(i).setInputs(ph(i-1), ef(i-1));
            
            m = ef(i-1).m;
            m2 = ef(i-1).m2;
           
            % Inicialización de entradas dependiendo de configuración       
            if i == 3*m+2 % [5,8,11,14] 
                % No entra destilado del efecto anterior: Md_in = 0, Td_in = 0
                % No entra destilado del mixer: Mmix_in = 0, Tmix_in = 0
                ef(i).Md_in = 0;
                ef(i).Td_in = 0;
                ef(i).Mda = 0;
                ef(i).Mdb = 0;
                ef(i).Tmix_in = 0;
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
            
            %--------------------------------------------------------------
            % NUEVO MIXER: vh_in, delta, d_in, mix_in -> dest_in (M, T)
            % Obtener calores específicos
            ef(i) = ef(i).calculoScpHeat();
            % Balance de masa:
            ef(i).Mdest = ef(i).Mvh + ef(i).Mdelta + ef(i).Md_in + ef(i).Mda;
            % Balance de energía:
            ef(i).Tdest = ( ef(i).Mvh * ef(i).Tv_in * ef(i).Cp_ph +...
                            ef(i).Mdelta * ef(i).Tv_in * ef(i).Cp_delta +...
                            ef(i).Md_in * ef(i).Td_in * ef(i).Cp_d +...
                            ef(i).Mda * ef(i).Tmix_in * ef(i).Cp_mix ) /...
                            ( ef(i).Mdest * ef(i).Cp_ph );
            %--------------------------------------------------------------
            
            ef(i).m = m;
            ef(i).m2 = m2;
            
            % Cálculos del efecto
%             x0 = [1 2 3 4 5];
            x0 = 0.1:1:5;
            A = []; b = []; Aeq = []; beq = []; 
            lb = 0.1; ub = 10;
            fun = @(U)calculoUef(U, Tv_ref(i),i,ef(i));

            for idx=1:length(x0)
                try
                    Uff(idx) = fmincon(fun,x0(idx),A,b,Aeq,beq,lb,ub,[],options_fmincon);
                catch
                    continue
                end
            end

            % Filter out repetead solutions
            idx = [true abs(diff(Uff))>0.01]; if ~any(idx), Uff=Uff(1); else, Uff=Uff(idx); end

            % And evaluate the output for each solution
            x0 = [ef(i).Mb_in*0.1,ef(i).Mb_in,ef(i).Xb_in,ef(i).Tv_in-2.7,...
                 ef(i).Tb_in-2.7,0,ef(i).Mb_in,ef(i).Xb_in,ef(i).Tb_in-2.7,...
                 ef(i).Mv_in, ef(i).Mdest, 0, ef(i).Tdest, ef(i).Tdest-2.7];

            fval = []; a_temp = [];
            for idx=1:length(Uff)
                a_temp(idx,:) = fsolve(@(x) efecto_i_flash(x,ef(i), Uff(idx)), x0, options_fsolve);
                fval(idx)       = abs( Tv_ref(i) - a_temp(idx, 4) );
            end
            
            % The correct solution is the one that returns the minimum value
            ef(i).Uef = Uff(fval==min(fval));
            a =      a_temp(fval==min(fval),:);

            [ef(i).Mgb, ef(i).Mb_out, ef(i).Xb_out, ef(i).Tv_out, ef(i).Tbb,...
            ef(i).Mgf, ef(i).Mbb, ef(i).Xbb, ef(i).Tb_out, ef(i).Mv_in,...
            ef(i).Mdest_in, ef(i).Mdest_f, ef(i).Tdest_in, ef(i).Tdest_out] = ...
            deal(a(1),a(2),a(3),a(4),a(5),a(6),a(7),a(8),a(9),a(10),a(11),a(12),a(13),a(14));

            error_ = abs(ef(i).Tv_out - Tv_ref(i));
            if error_ > 0.1
                error('calibrationModel:effectErrorThreshold', ...
                    ['The difference between the reference effect tempererature ' ...
                    'and the calculated one is too high (%.2f) for cell %d'], error_,i)
            end
            %  -------------------------- log ------------------------------------------------------------------------------ 
            if logging
            fprintf('\n_________________________________________ CELDA %2d ___________________________________________\n',i)
            fprintf('\nEF | Error = %4.2f [ºC]  |  Tref=%4.2f,    T=%4.2f [ºC]  |  U=%5.2f [kW/m2ºC]\n', ...
                error_, Tv_ref(i), ef(i).Tv_out, ef(i).Uef);            
            end
            % --------------------------------------------------------------------------------------------------------------

            ef(i) = ef(i).calculoScpHeat; %#ok<*AGROW>
            
            ef(i).Md_out = ef(i).Mdest_in + ef(i).Mv_in;

            ef(i).Td_out = ( ef(i).Mv_in * ef(i).Tv_in * cpSatLiqTW(ef(i).Tv_in) + ...
            ef(i).Mdest_in * cpSatLiqTW(ef(i).Tdest_out) * ef(i).Tdest_out )...
            / ( ef(i).Md_out * ( cpSatLiqTW(ef(i).Tv_in) + cpSatLiqTW(ef(i).Tdest_out) )/2 );

            if ef(i).Mda > 0
                ef(i).Mmix_out = ef(i).Mdb + ef(i).Md_out;
                ef(i).Tmix_out = ( ef(i).Mdb * ef(i).Tmix_in * ef(i).Cp_mix + ...
                    ef(i).Md_out * ef(i).Td_out * cpSatLiqTW(ef(i).Td_out) )...
                    / (ef(i).Mmix_out * ef(i).Cp_d);
            else
                ef(i).Mmix_out = 0;
                ef(i).Tmix_out = 0;
            end
            
            % Obtain energy input for each source
            ef(i) = ef(i).estimacionAportes2;

            % ADd boiling point elevation
            ef(i).bpe = SW_BPE(ef(i).Tb_out,ef(i).Xb_out);
            
            % Comprobar balance de materia en efecto
            if abs(ef(i).Mgf + ef(i).Mgb + ef(i).Mb_out - ef(i).Mb_in) > 0.0001
                warning("No se cumple balance de materia: Mb_in != Mgb + Mgf + Mb_out (%.2f != %.2f)",...
                    ef(i).Mb_in*3600, (ef(i).Mgf+ef(i).Mgb+ef(i).Mb_out)*3600)
            end
            
% -------------------------------------------------------------------------
%%                           PRECALENTADOR i                             %%
%--------------------------------------------------------------------------
        if i<Nef
            % Inicialización de entradas
            ph(i) = ph(i).setInputs(ef(i));
            ph(i).Tph_out = ph(i-1).Tph_in;
            ph(i).Cp = SW_SpcHeat(ph(i).Tph_out, ph(i).Xf)/1000;

            % Cálculos del precalentador
%                 x0 = [0.01 0.4 1.2 1.6 2 4];

            if ph(i).Tv < ph(i).Tph_out
                error('calibrationModel:invertedHE',['Error in preheater' ...
                    '%d, steam temperature in preheater (Tv) is smaller' ...
                    ' than feed water ouput temperature (Tph_out): ' ...
                    '%.2f < %.2f'], i, ph(i).Tv, ph(i).Tph_out)
            end

            x0 = 0.1:10;
            A = []; b = []; Aeq = []; beq = []; 
            lb = 0.1; ub = 20;
            fun = @(U)calculoUph_cls(U,Tph_ref(i+1),i,Nef,ph(i));

            Uff = [];
            for idx=1:length(x0)
                Uff(idx) = fmincon(fun,x0(idx),A,b,Aeq,beq,lb,ub,[],options_fmincon);                
            end             
            
            % Filter out repetead solutions
            idx = [true abs(diff(Uff))>0.01]; if ~any(idx), Uff=Uff(1); else, Uff=Uff(idx); end

            % And evaluate the output value
            fval = [];
            for idx=1:length(Uff)
                fval(idx) = fun(Uff(idx));
            end
            % The correct solution is the one that returns the minimum value
            ph(i).Uph = Uff(fval==min(fval));
            
            ph(i).Tph_in = fsolve(@(Tin) ...
            precalentador(Tin, ph(i).Uph, ph(i).Aph, ph(i).Tv, ...
            ph(i).Tph_out, ph(i).Mf, ph(i).Cp), ph(i).Tph_out-5, options_fsolve);

            error_ = abs(ph(i).Tph_in - Tph_ref(i+1));
            % ------------------------------------- log ----------------------------------------------------------------- 
            if logging
            fprintf('PH | Error = %4.2f [ºC]  |   Tin=%4.2f, Tout=%4.2f [ºC]  |  U=%5.2f [kW/m2ºC] | Tv=%5.2f [ºC]\n', ...
                error_, ph(i).Tph_in, ph(i).Tph_out, ph(i).Uph, ph(i).Tv);
            end
            % -----------------------------------------------------------------------------------------------------------

            if error_ > 0.1
                if debug, evaluacionPh(in, i, ph), end

                error('calibrationModel:preheaterErrorThreshold', ...
                      ['The difference between the reference preheater tempererature ' ...
                      'and the calculated one is too high (%.2f ºC) for cell %d'], error_,i)
            end

            ph(i).Mvh = ( ph(i).Mf * ph(i).Cp * (ph(i).Tph_out-ph(i).Tph_in) ) / ( enthalpySatVapTW(ph(i).Tv)-enthalpySatLiqTW(ph(i).Tv) );
            ph(i).Mv = ph(i).Mgb + ph(i).Mgf - ph(i).Mvh; %#ok<*SAGROW>

            if gfc; preheaters_evolution(t,i,ph(i),Tph_ref(i+1),ef(i), ef, ph); end


% -------------------------------------------------------------------------
%%                             CONDENSADOR                               %%
%--------------------------------------------------------------------------
        elseif i==Nef

            % Inicialización de entradas
            ph(Nef).Aph = Aph(Nef);
            ph(Nef) = ph(Nef).setInputs(ef(Nef));
%                 ph(Nef).Tph_out = ph(Nef-1).Tph_in;
            ph(Nef).Tph_out = in.Tcwout;
            ph(Nef).Tph_in = in.Tcwin;
            ph(Nef).Cp = SW_SpcHeat(ph(Nef).Tph_out, ph(Nef).Xf)/1000;
            
            % Cálculos del condensador
            % MIXER 14
            ef(Nef).Mmix_out = ef(Nef-1).Mmix_out + ef(Nef).Md_out; 
            ef(Nef).Tmix_out = ( ef(Nef-1).Mmix_out * ef(Nef-1).Tmix_out * cpSatLiqTW(ef(Nef-1).Tmix_out) + ...
                ef(Nef).Md_out * ef(Nef).Td_out * cpSatLiqTW(ef(Nef).Td_out) )...
                / (ef(Nef).Mmix_out * ef(Nef-1).Cp_mix);
            ef(Nef).Cp_mix = cpSatLiqTW(ef(Nef).Tmix_out);
            
            Mprod = ef(Nef).Mgf + ef(Nef).Mgb + ef(Nef).Mmix_out;
            Tprod = in.Tprod;
            ph(Nef).Mvh = Mprod;
%                 Tcw_out = in.Tcwout;
            
            Mdf = ef(Nef).Mmix_out * ef(Nef).Cp_mix * (ef(Nef).Tmix_out - ph(Nef).Tv) / ...
                ( enthalpySatVapTW(ph(Nef).Tv)-enthalpySatLiqTW(ph(Nef).Tv) );
            ph(Nef).Mdf = Mdf;

%                 x0 = 2;
%                 A = []; b = []; Aeq = []; beq = []; lb = 0.1; ub = 10;

%                 ph(Nef).Uph = fmincon(@(x) calculoUph_cls(x, in.Tcwin,i,Nef,ph(i)), x0,A,b,Aeq,beq,lb,ub,[],opt);
            if ph(Nef).Tv > in.Tcwin && ph(Nef).Tv > in.Tcwout
                ph(Nef).Uph = ( in.Msw * SW_SpcHeat(in.Tcwin, in.Xf)/1000 * ...
                    (in.Tcwout - in.Tcwin) * log( (ph(Nef).Tv - in.Tcwin)/(ph(Nef).Tv - in.Tcwout) ) ) /...
                    ( ph(Nef).Aph * ((ph(Nef).Tv - in.Tcwin)-(ph(Nef).Tv - in.Tcwout)) );
            else
                warning("Tv (%.2f) < Tcwout (%.2f) o Tcwin (%.2f)",...
                    ph(Nef).Tv, in.Tcwout, in.Tcwin)
                ph(Nef).Uph = 0;
            end
            if Tprod < in.Tcwin
                warning("Los duendes atacan de nuevo, Tcwin (%.1f) > Tprod (%.1f)", in.Tcwin, Tprod)
            end

            if gfc, preheaters_evolution(t,i,0,0,ef(Nef), ef, ph); end
%                 Tcw_in = fsolve(@(Tcw_in) condensador(Tcw_in, ph(Nef).Uph, ph(Nef).Aph, ph(Nef).Tv, ...
%                     Tcw_out, ph(Nef).Mgb, ph(Nef).Mgf, Mdf), ph(Nef).Tph_out-5, options);
            

%                 Msw = ( ph(Nef).Mgb + ph(Nef).Mgf + Mdf ) * ( enthalpySatVapTW(ef(Nef).Tv_out)-enthalpySatLiqTW(ef(Nef).Tv_out) )...
%                     / ( ph(Nef).Cp*(Tcw_out-Tcw_in) );
%                 if Msw > Mf
%                     Mcw = Msw-Mf;   % Agua de refrigeración
%                 else
%                     warning("El caudal de alimentación es superior al necesario "...
%                          + "para alcanzar el equilibrio en el condensador: %.2f < %.2f. " + ...
%                          "Mf debería reducirse a %.2f [kg/s]", Msw, Mf, Msw)
%                     Mcw = 0;
%                     Msw = Mf;
%                 end
            if abs(Mf - Mprod - ef(Nef).Mb_out) > 0.01  % ¿Qué diferencia es aceptable? ¿0?
                warning("No se cumple balance de materia: Mf != Mb + Mprod (%.2f != %.2f)",...
                    Mf*3600, (Mprod+ef(Nef).Mb_out)*3600)
            end

             % log
            if logging
             fprintf(  'PH |                       Tin=%4.2f,  Tout=%4.2f [ºC]  |  U=%5.2f [kW/m2ºC]  |  Tv=%4.2f [ºC]\n', ...
                ph(i).Tph_in, ph(i).Tph_out, ph(i).Uph, ef(i).Tv_out);
            fprintf('____________________________________________________________________________________________\n')
            end

            % STEC
            if strcmp(in.source, 'liquid')
                PR = Mprod * 2326 / (Ms * (in.Ts_in-in.Ts_out) * ef(1).Cp_s); 
            else
                PR = Mprod * 2326 / (Ms * (enthalpySatVapTW(Ts_in)-enthalpySatLiqTW(Ts_in)));
            end
            STEC = 2326/PR * densSatLiqTW(Tprod) * 1e-3; % [MJ/m3]
            % RR
            out.RR = Mprod/Mf;
            
            out.Mprod = Mprod;
            out.STEC = STEC; out.PR = PR; 

            % Check error in Mprod
            if logging && isfield(in, "Mprod")
            fprintf('<strong>Mprod_ref = %.2f | Mprod_model = %.2f   | Error =  %.2f   |   (kg/s)</strong>\n', ...
                    in.Mprod, out.Mprod, in.Mprod-out.Mprod)
            end

            % Results visualization
%             if gfc
%                 ef_input_energy_breakdown(ef, in)
%                 results_visualization
%             end
        end
        end
    end
    ef = ef(i_inicial:i_final);
    ph = ph(i_inicial:i_final);
end


function preheaters_evolution(t,k,ph,Tph_ref,ef, efs, phs)
    % Temperatures evolution
    if k<14
    ax = nexttile(t, [1 2]); hold(ax, "on")
    if k==1; title(ax, 'Temperature evolution'); end
    plot([ph.Tv ph.Tv], [0 1]);
    
    plot([ph.Tph_in ph.Tph_out],[0 1], '-o')
    errorbar(abs(ph.Tph_in-Tph_ref), 1, 1,'horizontal', 'Color','#D95319')
    
    if k<13
        text(ph.Tph_in-0.5,0,    sprintf('Tph(%d)',k+1))
    else
        text(ph.Tph_in-0.5,0+0.2,    sprintf('Tph(%d)',k+1))
    end
    text(ph.Tph_out-0.5,  1, sprintf('Tph(%d)',k))
    text(ph.Tv+0.5, 0.5,     sprintf('Tv(%d)',k), HorizontalAlignment="center", Rotation=90)
    
    ax.XDir='reverse'; ax.XLim = [20 75];
    ax.TickDir = 'in'; ylabel(ax,string(k), Rotation=0); ax.XTick = []; %NONE
    ax.YTick = []; box(ax,'off'); ax.XColor = 'white';
%     ax.YAxisLocation = 'left';
    if k==13; xlabel(ax, '[ºC]'); ax.XTick=25:5:75; ax.XColor = 'black';end
    end

    % Heat transfer coefficients evolution plot
    if k>1
    ax = nexttile(t); hold(ax, "on")
    ax.YDir = 'reverse';

        
%         plot(ef.Uef, k, '-o', MarkerFaceColor='#0072BD')
%         plot(ph.Uph, k,'-o', MarkerFaceColor='#D95319');
%         ax.YTick = k-1; 
%     else
        plot([efs(k-1).Uef ef.Uef], [k-1 k], '-o', MarkerFaceColor='#0072BD')
        plot([phs(k-1).Uph phs(k).Uph], [k-1 k],'-o',  MarkerFaceColor='#D95319');
        
%     end
    
    % Text label positioning
    xoffset = 1;
    if k==2, yoffset = 0.2; else, yoffset=0; end

    if efs(k-1).Uef < phs(k-1).Uph, xoffset=xoffset*-1; end
    text(efs(k-1).Uef+xoffset, k-1+yoffset, sprintf('%.1f',efs(k-1).Uef), BackgroundColor='#0072BD', Color='w', HorizontalAlignment='center')
    text(phs(k-1).Uph-xoffset, k-1+yoffset, sprintf('%.1f',phs(k-1).Uph), BackgroundColor='#D95319', Color='w', HorizontalAlignment='center')

%     legend(sprintf('ef:%.2f',ef.Uef), sprintf('ph:%.2f',ph.Uph), "Orientation","horizontal")
%     text(ph.Uph, k, sprintf('%3.2f',ph.Uph), HorizontalAlignment="left")
%     text(ef.Uef, k, sprintf('%3.2f',ef.Uef), HorizontalAlignment="right")

    if rem(k,2)
        ax.YTick = [k-1 k]; ax.YAxisLocation = 'left'; 
    else
        ax.YTick = [];
    end
%     ax.XLim = [0 7]; 
    ax.XTick = []; ax.TickDir = 'in'; %NONE
    box(ax,'off'); ax.XColor = 'white';
    ax.XLim = [-2 8];
    end

    if k==2; title(ax, 'Heat tranfer coefs'); end
    if k==14
        text(efs(k).Uef+xoffset, k-0.2, sprintf('%.1f',efs(k).Uef), BackgroundColor='#0072BD', Color='w', HorizontalAlignment='center')
        text(phs(k).Uph-xoffset, k-0.2, sprintf('%.1f',phs(k).Uph), BackgroundColor='#D95319', Color='w', HorizontalAlignment='center')
        xlabel(ax, '[KW/ºC·m²]'); 
        legend('Uef', 'Uph')
        ax.XColor = 'black';
        ax.XTick = 0:2:10;
%         nexttile('south')
%         l = legend('Tv', 'Tph', 'Uph', 'Uef');
%         l.Layout.Tile('south');
    end
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


% function heatTransferCoef_evolution(t,ph,ef)
%     nexttile(t, [14 1])
%     plot([ph.Uph], 1:14);
%     hold on
%     plot([ef.Uef], 1:14);
%     
%     ax = gca; ax.YAxisLocation='right';
%     xlabel(ax,'[W/m2]')
% end
