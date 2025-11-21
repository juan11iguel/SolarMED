classdef Preheater2
    
    % Clase que almacena las variables de interés de un precalentador
    % cualquiera de una planta desaladora tipo MED.
    %
    % Escibir Precalentador.{nombre propiedad} o 
    % Precalentador.{nombre método} para obtener más información de la
    % propiedad o método concreto.

   properties
   % Inputs
        % Tph_out. Temperatura con la que el agua de alimentación abandona el
        % precalentador [ºC]
        Tph_out(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} 
        % Mf. Flujo másico de agua de alimentación [kg/s]
        Mf(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Xf. Concentración de sales de agua de alimentación [g/kg]
        Xf(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Tv. Temperatura de vapor proveniente del efecto anterior [ºC]
        Tv(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Mv. Flujo másico de vapor que abandona el precalentador [kg/s]
        Mv_in(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        
        % Específicas de condensador
        % Mcw. Flujo másico de agua de refrigeración [kg/s]
        Mcw(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}=0
        % Mmix_in. Flujo másico de destilados que entra en el condensador
        % procedentes de la línea de distribución
        Mmix_in(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}=0
        % Tmix_in. Temperatura de destilados que entra en el condensador
        % procedentes de la línea de distribución
        Tmix_in(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}=0

    % Outputs
        % Tph_in. Temperatura con la que el agua de alimentación entra en el
        % precalentador [ºC]
        Tph_in(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Mv_out. Fujo másico de vapor que abandona el precalentador
        Mv_out(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Mvh. Flujo de destilado que abandona el precalentador [kg/s] 
        Mvh(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Tvh. Temperatura de destilado que abandona el precalentador [kg/s]
        Tvh(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Mcw_no_over_cooling. Flujo másico de agua de refrigeración
        % necesario para condensar vapor en condensador (sin subenfriar) [kg/s]
        Mcw_no_over_cooling(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}

    % Internal variables
        % Distillate that enters the condenser and flashes. [kg/s]
        Mdf(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}

    % Parameters
        % Coeficiente de transferencia de valor del efecto. [kW/m2ºC]
        Uph(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
        % Área de intercambio del precalentador [m2]
        Aph(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 5
        % Cp. Calor específico de la alimentación para las condiciones dadas
        % [KJ/kg K]
        Cp(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 4.18
        % Cp_mix. Calor específico del destilado proveniente del mezclador 
        % [KJ/kg K]
        Cp_mix(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
   end
 

end