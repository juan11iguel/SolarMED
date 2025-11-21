classdef Preheater
    
    % Clase que almacena las variables de interés de un precalentador
    % cualquiera de una planta desaladora tipo MED.
    %
    % Escibir Precalentador.{nombre propiedad} o 
    % Precalentador.{nombre método} para obtener más información de la
    % propiedad o método concreto.
    %
    % Métodos de Precalentador:
    %   calcularPerdidasTuberia_PH-EF - Estima pérdidas en tubería que
    %   recorre desde salida del precalentador hasta entrada de efecto

   properties
      % Mvh. Flujo de destilado que abandona el precalentador [kg/s] 
      Mvh(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} 
      % Tph_in. Temperatura con la que el agua de alimentación abandona el
      % precalentador [ºC]
      Tph_in(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
      % Tph_out. Temperatura con la que el agua de alimentación entra en el
      % precalentador [ºC]
      Tph_out(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} 
      % Mf. Flujo másico de agua de alimentación [kg/s]
      Mf(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
      % Xf. Concentración de sales de agua de alimentación [g/kg]
      Xf(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
      % Tv_in. Temperatura de vapor proveniente del efecto anterior [ºC]
      Tv(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
      % Mv. Flujo másico de vapor que abandona el precalentador [kg/s]
      Mv(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
      % Mgf. Flujo másico de vapor generado por flash [kg/s]
      Mgf(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
      % Mgb. Flujo másico de vapor generado por evaporación [kg/s]
      Mgb(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
      % Coeficiente de transferencia de valor del efecto. [kW/m2ºC]
      Uph(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
      % Pérdida de vapor en el efecto debido a condensación [kg/s]
      Ltub(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 0.7 
      % Diámetro del tramo de tubería que une precalentador y efecto [mm]
      Dtub(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 200
      % Área de intercambio del efecto [m2]
      Aph(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 5
      % Mv_out. Fujo másico de vapor que abandona el precalentador
            % alto del demister [m]
      h_dem(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 0.13
      % ancho del demister [m]
      w_dem(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 5e-2
      % largo del demister [m]
      l_dem(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 1.97
      % densidad del demister [m]
      rho_dem(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 200
      % diámetro de los cables que forman el demister [mm]
      D_dem(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 0.28
      % Cp. Calor específico de la alimentación para las condiciones dadas
      % [KJ/kg K]
      Cp(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative} = 4.18
      % Mdf. [kg/s]
      Mdf(1,1) double {mustBeReal, mustBeFinite,mustBeNonnegative}
   end
   methods
       function Tvv = calculoPerdidasDemister(obj)
       % Método que calcula las pérdidas de carga en el demister en base a
       % El-Dessouky et al. (2000), parámetros:
       % - v_vap: Velocidad del vapor a la entrada del demister [m/s]
       % - l_dem: largo del demister [m]
       % - h_dem: alto del demister [m]
       % - w_dem: ancho del demister [m]
       % - rho_dem: densidad del demister [kg/m3]
       % https://www.demisterpads.com/technology/stainless-steel-demister-pad.html
       % (ejemplos de demister industriales)
      
           v_vap = obj.Mgb/(densSatVapTW(obj.Tv)*obj.l_dem*obj.h_dem);
           deltaPdem = ( 3.88178*(obj.rho_dem)^0.375798 * (v_vap)^0.81317 * (obj.D_dem)^-1.56114147 ) * obj.w_dem; % [Pa]
           deltaPdem = deltaPdem/1e5; % [bar]

           Tvv = tSatW( pSatW(obj.Tv)-deltaPdem ); 
       end
       
       function obj = setInputs(obj, ef)
       % Método que establece las entradas necesarias para los cálculos del
       % precalentador a partir de las salidas de interés del efecto
       % contiguo. Entradas:
       % ef: objeto de la clase tipo Efecto
           obj.Mgb = ef.Mgb;
           obj.Mgf = ef.Mgf;
           obj.Tv = ef.Tv_out;
           obj.Tv = obj.calculoPerdidasDemister;
       end
   end
end