function evaluacionPh(in, i, ph)
%% Evaluación precalentador
%% Ejecutar desde interrupcion
% i: precalentador a evaluar 
gris        = [0.25, 0.25, 0.25]; %#ok<*NASGU> 
verde       = [0.4660, 0.6740, 0.1880];
rojo        = [0.6350, 0.0780, 0.1840];
azul        = [0, 0.4470, 0.7410];
azul_claro  = [0.3010, 0.7450, 0.9330];
naranja     = '#D95319';

% cp = SW_SpcHeat(in.Tph_ref(i+1), 3.3062)/1000;
cp = ph(i).Cp;

rangoU = 0.001:0.005:20; j=1; Tph= zeros(length(rangoU),1);
for Uph=rangoU
    
    Tph(j) = precalentador(in.Tph_ref(i+1), Uph, in.Aph(i), ...
             ph(i).Tv, in.Tph_ref(i), in.Mf, cp) + in.Tph_ref(i+1);
    j=j+1;
end

idx = min( abs(Tph-in.Tph_ref(i+1)) ) == abs(Tph-in.Tph_ref(i+1));
interseccion = rangoU( idx );
Tph_manual = Tph(idx);

% options = optimoptions('fsolve','FunctionTolerance',1e-9, ...
%                        'OptimalityTolerance',1e-12, ...
%                        'StepTolerance',1e-12, ...
%                        'Algorithm','levenberg-marquardt', ...
%                        'PlotFcn','optimplotfval')
options = [];

Tph_fsolve = fsolve(@(Tin) precalentador(Tin, interseccion, ...
             in.Aph(i), ph(i).Tv, in.Tph_ref(i), in.Mf, cp), ...
             in.Tph_ref(i)-10, options);

fprintf('Debug: %.2f  |  fsolve: %.2f  [ºC],  U=%.2f [kW/m2ºC]\n', ...
         Tph_manual, Tph_fsolve, interseccion)

% Plot results
figure('Units','normalized','Position',[0 0 0.5 1])

plot(rangoU, Tph, LineWidth=2, Color=naranja); 
hold on

yl = yline(in.Tph_ref(i+1), '-', ...
     {"Tph\_in\_ref="+string(in.Tph_ref(i+1))}, ...
     Color=[0.3010, 0.7450, 0.9330], LineWidth=2);   % Tin_ref
yl.LabelVerticalAlignment = 'bottom';

yl = yline(in.Tph_ref(i), '-', {"Tph\_out\_ref="+string(in.Tph_ref(i))}, ...
     Color=[0, 0.4470, 0.7410], LineWidth=2);        % Tout_ref

yl.LabelVerticalAlignment = 'bottom';
yline(ph(i).Tv, '-', {"Tv\_ref="+string(ph(i).Tv)}, ...
    Color=[0.4940, 0.1840, 0.5560], LineWidth=2);    % Tv_ref

% xl = xline(interseccion, '-', {"Uph="+string(interseccion)}, LineWidth=2);            % Uph
% xl.LabelVerticalAlignment='bottom'; xl.LabelHorizontalAlignment='left';xlabel('Uph'); ylabel('Tin');

% Results
plot([interseccion interseccion], [Tph_fsolve Tph_manual],'x',MarkerEdgeColor='red',MarkerSize=15, LineWidth=4)

legend('Tph\_calculada', 'Tph\_in\_ref','Tph\_out\_ref', 'Tv\_ref','',Location='best')
title('Heat transfer coefficient evaluation for preheater', sprintf('Uph=%.2f [KW/m2ºC]',interseccion))
xlim([interseccion-1.5 interseccion+1.5])
grid on; grid minor

end