function pts = filterPts(pts, varargin)
% [pts, pts_original] = loadData(pts_original, load_from_db=false);

iP = inputParser;
check_logical = @(x)validateattributes(x, {'logical'},{'scalar'});
% addParameter(iP, 'save_results', false, check_logical)
addParameter(iP, 'gfc', false, check_logical)

parse(iP,varargin{:})

% save_results = iP.Results.save_results;
gfc = iP.Results.gfc;

Lpts0 = length(pts);

deleted=true; cont = 0;
while deleted
    deleted=false;

    notCheckVars = "";

    check_deltaTs_in = abs(diff( [vertcat(vertcat(pts.medidas).Ts_in).value])) > 0.5;  % ºC
    if any(~check_deltaTs_in), notCheckVars = [notCheckVars; "ΔTs_in < 0.5"]; end
    
    check_deltaTs_out= abs(diff( [vertcat(vertcat(pts.medidas).Ts_out).value])) > 0.5; % ºC
    if any(~check_deltaTs_out), notCheckVars = [notCheckVars; "ΔTs_out < 0.5"]; end
    
    check_deltaTvc   = abs(diff( [vertcat(vertcat(pts.medidas).Tvc).value])) > 1.5;    % mbar
    if any(~check_deltaTvc), notCheckVars = [notCheckVars; "ΔTvc < 1.5"]; end
    
    check_deltaMs    = abs(diff( [vertcat(vertcat(pts.medidas).Ms).value])) > 0.2;     % L/s
    if any(~check_deltaMs), notCheckVars = [notCheckVars; "ΔMs < 0.2"]; end
    
    check_deltaMf    = abs(diff( [vertcat(vertcat(pts.medidas).Mf).value])) > 0.2;     % m³/h
    if any(~check_deltaMf), notCheckVars = [notCheckVars; "ΔMf < 0.2"]; end
    
    check_deltaTcwout= abs(diff( [vertcat(vertcat(pts.medidas).Tcwout).value])) > 0.5; % ºC
    if any(~check_deltaTcwout), notCheckVars = [notCheckVars; "ΔTcwout < 0.5"]; end

    op_time = [vertcat(vertcat(pts.operatedTime).effects).timeOperated]; op_time = op_time(1:14:end);
    check_tiempoOperado      = abs(diff( op_time )) > hours(10); % hours operated
    if any(~check_tiempoOperado), notCheckVars = [notCheckVars; "ΔOperated time < 10 hours"]; end

    check_tiempoTranscurrido = abs(hours(diff( [pts.time]))) > hours(120); % 1 semana
    if any(~check_tiempoTranscurrido), notCheckVars = [notCheckVars; "Elapsed time < 1 week"]; end
    
    check = check_deltaTs_in | check_deltaTs_out | check_deltaTvc | ...
            check_deltaMs    | check_deltaMf     | check_deltaTcwout | ...
            check_tiempoOperado | check_tiempoTranscurrido;

    if any( ~check )
        pts( ~check ) = [];

        fprintf('Deleted %d points, non compliant constrains for: ', sum(~check))
        for i=2:length(notCheckVars), fprintf('%s, ', notCheckVars(i)), end
        fprintf('\n');

        cont = cont + sum(~check);
        deleted=true;
    end

end

title_str = sprintf('Deleted %d out of %d operation points for being too similar to consecutive operation points\n', cont, Lpts0);
disp(title_str) %#ok<*DSPS> 

if gfc
    figure
    tl = tiledlayout(2,1,"TileSpacing","compact");
    repOpPtsMap([], pts, background_pts=pts_original, custom_color=[0.4660, 0.6740, 0.1880]);
    
    ax = nexttile(tl); hold(ax, "on");
    medidas = [pts_original.medidas]; var = [medidas.Ts_in];
    gris  = gray(10);
    
    % Ptos en base de datos
    scatter(ax, [pts_original.time], [var.value], 20,'filled', ...
           'MarkerEdgeColor', gris(4,:),...
           'MarkerFaceColor', gris(3,:),...
           'MarkerEdgeAlpha', .1, ...
           'MarkerFaceAlpha', .05);
    scatter(ax, [pts.time], [vertcat(vertcat(pts.medidas).Ts_in).value], 30,[0.4660, 0.6740, 0.1880], 'filled')
    ylabel(ax, 'Ts,in (ºC)')
    title(ax, 'Timeline')
    
    lg = legend(ax, 'Dataset from steady state detection', 'Filtered points', Orientation='horizontal');
    lg.Layout.Tile = 'South';
    title(tl, title_str)
end

end