function Tv_out = demister(Mv, Tv_in)
    % Método que calcula las pérdidas de carga en el demister en base a
    % El-Dessouky et al. (2000), parámetros:
    % - v_vap: Velocidad del vapor a la entrada del demister [m/s]
    % - l_dem: largo del demister [m]
    % - h_dem: alto del demister [m]
    % - w_dem: ancho del demister [m]
    % - rho_dem: densidad del demister [kg/m3]
    % https://www.demisterpads.com/technology/stainless-steel-demister-pad.html
    % (ejemplos de demister industriales)

    % altura del demister [m]
    h_dem = 0.13;
    % ancho del demister [m]
    w_dem = 5e-2;
    % largo del demister [m]
    l_dem = 1.97;
    % densidad del demister [m]
    rho_dem = 200;
    % diámetro de los cables que forman el demister [mm]
    D_dem = 0.28;

   v_vap = Mv/(densSatVapTW(Tv_in)*l_dem*h_dem);
   deltaPdem = ( 3.88178*(rho_dem)^0.375798 * (v_vap)^0.81317 * (D_dem)^-1.56114147 ) * w_dem; % [Pa]
   deltaPdem = deltaPdem/1e5; % [bar]

   Tv_out = tSatW( pSatW(Tv_in)-deltaPdem ); 
end