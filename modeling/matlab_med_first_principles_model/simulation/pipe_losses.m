function Tv_out = pipe_losses(Mv, Tv_in)
    % Método que calcula las pérdidas de carga en la tubería en base a la 
    % expresión de Unwin, parámetros:
    % - Ltub: Longitud de la tubería [m]
    % - Dtub: Diámetro interno de la tubería [mm]
    % - w: Flujo másico de vapor [kg/s]
    % - Tv_in: Temperatura del vapor al inicio del recorrido [ºC] 

    % Longitud del tramo de tubería que une precalentador y efecto [m]
    Ltub = 0.7;
    % Diámetro del tramo de tubería que une precalentador y efecto [mm]
    Dtub = 200;
    
    
    deltaPcl = ( 0.6753e6*(Mv*3600)^2*Ltub*(1 + 91.4/Dtub) ) /...
               ( densSatVapTW(Tv_in)*Dtub^5 ); % [Pa]
    deltaPcl = deltaPcl/1e5; % [bar]
    Tv_out = tSatW( pSatW(Tv_in)-deltaPcl );
end