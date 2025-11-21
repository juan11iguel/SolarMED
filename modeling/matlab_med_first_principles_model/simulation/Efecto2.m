classdef Efecto2
    % Clase que almacena las variables de interés de un efecto
    % cualquiera de una planta desaladora tipo MED.
    %
    % Escibir Efecto.{nombre propiedad} o Efecto.{nombre método} para
    % obtener más información de la propiedad o método concreto.
    %
    % Métodos de Efecto:
    %
    %   estimacionAportes - Calcula y almacena los aportes en la generación
    %   de vapor del efecto
    %
   properties

    % Inputs
        % Vapour
        % Mv_in. Flujo másico de vapor proveniente del precalentador anterior [kg/s]
        Mv_in(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 0
        % Tv_in. Temperatura de vapor proveniente del efecto anterior [ºC]
        Tv_in(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 0

        % Condensates
        % Mvh. Flujo de destilado que proviene del precalentador anterior [kg/s] 
        Mvh(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 0
        % Md_in. Flujo de destilado proveniente del efecto anterior [kg/s]
        Md_in(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 0
        % Td_in. Temperatura del destilado proveniente del efecto anterior [ºC]
        Td_in(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 0
        % Mda. Fracción de destilado proveniente del mezclador que se 
        % introduce en el efecto [kg/s]
        Mda(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 0
        % Mdb. Fracción destilado proveniente del mezclador y circula 
        % directamente al siguiente mezclador [kg/s]
        Mdb(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 0
        % Tmix_in. Temperatura del destilado que entra procedente del mezclador de
        % la línea de distribución [ºC]
        Tmix_in(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 0
        % Flujo másico de vapor del efecto anterior que condensa entre salida del
        % condensador y entrada al efecto [kg/s]
        Mdelta(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 0
        % Mdest_in. Destilado total que entra en el efecto [kg/s]
        Mdest_in(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 0
        % Tdest_in. Temperatura del destilado que entra en el efecto [ºC]
        Tdest_in(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 0
        
        % Feed
        % Fujo de salmuera proveniente del efecto anterior, o de la
        % alimentación si se trata del primer efecto (Mf) [kg/s]
        Mb_in(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 0
        % Xb_in. Concentración de salmuera proveniente del efecto anterior, o
        % de la alimentación si se trata del primer efecto (Xf) [g/kg]
        Xb_in(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 0
        % Tb_in. Temperatura con la que la salmuera entra en el efecto
        % proveniente del anterior, o temperatura del agua de alimentación si
        % se trata del primer efecto (Tf) [ºC]
        Tb_in(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 0
        
        % External
        % Ms. Flujo másico de fluido que aporta energía en el primer efecto
        % [kg/s]
        Ms(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 0
        % Ts_in. Temperatura con la que el fluido que aporta la energía entra
        % en el primer efecto. [ºC]
        Ts_in(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 0
        % Ts_out. Temperatura con la que el fluido que aporta la energía
        % abandona el primer efecto. [ºC] (liquid)
        Ts_out(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 0
        % Ps. Presión del fluido que aporta energía en el primer efecto
        % [bar] (liquid)
        Ps(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 0

        % Time operated
        % op_time. Tiempo operado desde la última limpieza del efecto
        op_time(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 0


    % Outputs
        % Condensates
        % Md_out. Flujo másico de destilado que abandona el efecto. [kg/s]
        Md_out(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Td_out. Temperatura con la que el destilado abandona el efecto. [ºC]
        Td_out(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % [bar]
        % Mmix_out. Destilado que abandona el mezclador de línea de 
        % distribución [kg/s]
        Mmix_out(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Tmix_out. Temperatura del destilado que abandona el mezclador de
        % línea de distribución [ºC]
        Tmix_out(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}

        % Brine
        % Flujo másico de salmuera que abandona el efecto. [kg/s]
        Mb_out(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Temperatura con la que la salmuera abandona el efecto. [ºC]
        Tb_out(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Concentración de la salmuera que abandona el efecto [g/kg]
        Xb_out(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}

        % Vapour
        % Flujo másico total de vapor que abandona el efecto [kg/s]
        Mv_out(1,1) double {mustBeReal, mustBeFinite, mustBeNonnegative}
        % Temperatura del vapor generado en el efecto. [ºC]
        Tv_out(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}


    % Internal variables
        % Vapour
        % Flujo másico de vapor generado por flash [kg/s]
        Mgf(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Flujo másico de vapor generado por evaporación [kg/s]
        Mgb(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Mdest_f. Destilado que flashea y se convierte en vapor [kg/s]
        Mdest_f(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}

        % Condensates
        % Mdest. Flujo másico del destilado producto de la mezcla de todos los
        % destilados después de entrar en el efecto [ºC]
        Mdest(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Tdest. Temperatura del destilado producto de la mezcla de todos los
        % destilados después de entrar en el efecto [ºC]
        Tdest(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Tdest_out. Temperatura de salida del destilado producto de la
        % mezcla de todos los destilados al salir del efecto
        Tdest_out(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}

        % Others
        % Temperatura del destilado proveniente del precalentador anterior
        % que abandona el efecto subenfriado. [ºC]
        Tvv(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Temperatura del destilado proveniente del efecto anterior
        % que abandona el efecto subenfriado. [ºC]
        Tdd(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Temperatura del destilado proveniente del mezclador
        % que abandona el efecto subenfriado. [ºC]
        Tmixx(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Temperatura de la salmuera que no flashea [ºC]
        Tbb(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Flujo másico de salmuera que no flashea [kg/s]
        Mbb(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Concentración de la salmuera que no flashea [g/kg]
        Xbb(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}

        % Specific heats
        % Cp_b. Calor específico de la salmuera proveniente del efecto
        % anterior [KJ/kg K]
        Cp_b(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Cp_ph. Calor específico del destilado proveniente del precalentador
        % anterior [KJ/kg K]
        Cp_ph(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Cp_d. Calor específico del destilado proveniente del efecto anterior 
        % [KJ/kg K]
        Cp_d(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Cp_mix. Calor específico del destilado proveniente del mezclador 
        % [KJ/kg K]
        Cp_mix(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Cp_mixx. Calor específico del destilado proveniente del mezclador
        % tras su paso por el efecto quedando subenfriado [KJ/kg K]
        Cp_mixx(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Cp_dd. Calor específico del destilado proveniente del efecto anterior 
        % tras su paso por el efecto quedando subenfriado [KJ/kg K]
        Cp_dd(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Cp_vv. Calor específico del destilado proveniente del precalentador 
        % anterior tras su paso por el efecto quedando subenfriado [KJ/kg K]
        Cp_vv(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Cp_delta. Calor específico del destilado proveniente del efecto anterior 
        % tras su condensación por el recorrido [KJ/kg K]
        Cp_delta(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Cp_s. Calor específico del fluido aportador de energía [KJ/kg K]
        Cp_s(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}

        % Energy
        % Aporte energético del vapor procedente de la celda anterior
        aporte_v_ant(1,1) double {mustBeReal, mustBeFinite}
        % Aporte energético del destilado procedente del precalentador
        % anterior
        aporte_ph_ant(1,1) double {mustBeReal, mustBeFinite}
        % Energía absorbida por calor sensible de salmuera que no evapora
        aporte_b(1,1) double {mustBeReal, mustBeFinite}
        % Aporte enerético del vapor condensado del efecto anterior
        aporte_ef_ant(1,1) double {mustBeReal, mustBeFinite}
        % Aporte energético del mixer de la línea de distribución
        aporte_mix(1,1) double {mustBeReal, mustBeFinite}
        % Aporte energético de la parte del vapor de la celda anterior que
        % condensa en el camino al efecto
        aporte_delta(1,1) double {mustBeReal, mustBeFinite}
        % Aporte energético total de la suma de destilados (aporte
        % energético por calor sensible)
        aporte_dest(1,1) double {mustBeReal, mustBeFinite} 
        % Aporte energético de una fuente de energía externa
        aporte_ext_source(1,1) double {mustBeReal, mustBeFinite} 


    % Parameters
        % Coeficiente de transferencia de valor del efecto. [kW/m2ºC]
        Uef(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Pérdida de vapor en el efecto debido a condensación [kg/s]
        deltaMv(1,1) double {mustBeReal,mustBeNonnegative,mustBeLessThanOrEqual(deltaMv,1)} = 1
        % Área de intercambio del efecto [m2]
        Aef(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 26.28
        % Proporción de destilado que abandona un efecto y se divide en:
        % efecto + 3 y mixer efecto + 3
        Y(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 0.01
        % m. Contador para distinguir efectos que no reciben destilado
        % proveniente del efecto anterior
        m(1,1) double {mustBeInteger, mustBeFinite,mustBeNonnegative} = 1
        % m2. Contador para distinguir efectos que reciben destilado
        % proveniente del mixer
        m2(1,1) double {mustBeInteger, mustBeFinite,mustBeNonnegative} = 1  
        % source. Estado físico en el que el fluido aportador de energía
        % entra en el primer efecto [liquid o steam]
        source(1,1) string
   end
   
   methods
       function obj = calculoScpHeat(obj)
          % Método que calcula los calores específicos y los almacena en la
          % clase para los distintos aportes
          if obj.Tb_out > 0
              obj.Cp_b  = SW_SpcHeat(obj.Tb_out,obj.Xb_out)/1000;
          else
              obj.Cp_b = 0;
          end
          if obj.Tv_in > 0
              obj.Cp_ph = cpSatLiqTW(obj.Tv_in);
              obj.Cp_delta = obj.Cp_ph;
          else
              obj.Cp_ph = 0;
              obj.Cp_delta = obj.Cp_ph;
          end
          if obj.Td_in > 0
              obj.Cp_d  = cpSatLiqTW(obj.Td_in);
          else
              obj.Cp_d = 0;
          end
          if obj.Tmix_in > 0
              obj.Cp_mix  = cpSatLiqTW(obj.Tmix_in);
          else
              obj.Cp_mix = 0;
          end 
          if obj.Tmixx > 0
              obj.Cp_mixx  = cpSatLiqTW(obj.Tmixx);
          else
              obj.Cp_mixx = 0;
          end 
          if obj.Tdd > 0
              obj.Cp_dd  = cpSatLiqTW(obj.Tdd);
          else
              obj.Cp_dd = 0;
          end 
          if obj.Tvv > 0
              obj.Cp_vv  = cpSatLiqTW(obj.Tvv);
          else
              obj.Cp_vv = 0;
          end 
       end
       
        function obj = estimacionAportes2(obj)
            % Método que estima el aporte en la generación de vapor en el efecto
            % por cada uno de los aportadores.,
            deltaH_Tviant  = enthalpySatVapTW(obj.Tv_in)   - enthalpySatLiqTW(obj.Tv_in);
            deltaH_ph      = enthalpySatLiqTW(obj.Tv_in)   - enthalpySatLiqTW(obj.Tv_out);
            deltaH_ef_ant  = enthalpySatLiqTW(obj.Td_in)   - enthalpySatLiqTW(obj.Tv_out);
            deltaH_mix     = enthalpySatLiqTW(obj.Tmix_in) - enthalpySatLiqTW(obj.Tv_out);
            
            Cp=4.18;
            
            Tvv =      obj.Tv_in   - deltaH_ph/Cp; %#ok<*PROP> 
            Tdd =      obj.Td_in   - deltaH_ef_ant/Cp;
            Tmixx =    obj.Tmix_in - deltaH_mix/Cp;
            
            % Proporción de cada aporte
            pvh = obj.Mvh/obj.Mdest;
            pef_ant = obj.Md_in/obj.Mdest;
            pmix = obj.Mda/obj.Mdest;
            pdelta = obj.Mdelta/obj.Mdest;
            
            % Ms
            if obj.Ms > 0, obj.aporte_ext_source = obj.Ms * Cp * (obj.Ts_in - obj.Ts_out);
            else, obj.aporte_ext_source = 0; end
            % Mv_in
            if obj.Mv_in > 0, obj.aporte_v_ant = obj.Mv_in*(deltaH_Tviant);
            else, obj.aporte_v_ant = 0; end
            % Mvh
            if obj.Mvh > 0, obj.aporte_ph_ant = (obj.Mvh-obj.Mdest_f*pvh)*Cp*(obj.Tv_in-Tvv);
            else, obj.aporte_ph_ant = 0; end
            % Mb_in
            if obj.Mb_in > 0, obj.aporte_b = obj.Mbb *Cp* (obj.Tb_out-obj.Tbb);
            else, obj.aporte_b = 0; end
            % Md_in
            if obj.Md_in > 0, obj.aporte_ef_ant = (obj.Md_in-obj.Mdest_f*pef_ant) *Cp*(obj.Td_in-Tdd);
            else, obj.aporte_ef_ant = 0; end
            % Mda (mixer)
            if obj.Mda > 0, obj.aporte_mix = (obj.Mda-obj.Mdest_f*pmix) *Cp*(obj.Tmix_in-Tmixx);
            else, obj.aporte_mix = 0; end
            % Mdelta
            if obj.Mdelta > 0, obj.aporte_delta = (obj.Mdelta-obj.Mdest_f*pdelta) *Cp*(obj.Tv_in-Tvv);
            else, obj.aporte_delta = 0; end
            
            obj.aporte_dest = obj.aporte_ph_ant + obj.aporte_ef_ant + obj.aporte_mix + obj.aporte_delta;
      end
      
      function obj = estimacionAportes(obj)
      % Método que estima el aporte en la generación de vapor en el efecto
      % por cada uno de los aportadores.
         deltaHTviant  = enthalpySatVapTW(obj.Tv_in) - enthalpySatLiqTW(obj.Tv_in);
         
         obj.aporte_v_ant =     obj.Mv_in       *(deltaHTviant);
         obj.aporte_ph_ant =    obj.Mvh    *Cp  *(obj.Tv_in-obj.Tvv);
         obj.aporte_b =         obj.Mbb    *Cp  *(obj.Tb_out-obj.Tbb);
         obj.aporte_ef_ant =    obj.Md_in  *Cp  *(obj.Td_in-obj.Tdd);
         obj.aporte_mix =       obj.Mda    *Cp  *(obj.Tmix_in-obj.Tmixx);
      end

      
   end
end