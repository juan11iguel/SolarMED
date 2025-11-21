function ph = preheater_simulation_block(ph, ann, varargin)
    
    iP = inputParser;    
    check_logical = @(x)validateattributes(x, {'logical'},{'scalar'});
    
    addParameter(iP, 'condenser', false, check_logical)
    addParameter(iP, 'ph_ref',  [])

    parse(iP,varargin{:})
    condenser = iP.Results.condenser;
    ph_ref  = iP.Results.ph_ref;
%     ph_idx    = iP.Results.ph_idx;

    options_fsolve = optimoptions('fsolve','Display','none', 'Algorithm','levenberg-marquardt');

    % Obtain Uph for the input variables
    inputs = [ph.Aph ph.Tph_out ph.Mf ph.Mcw ph.Mv_in ph.Tv];
    ph.Uph = evaluate_trained_ann(inputs, ann);
%     ph.Uph = ph_ref.Uph;

    % PREHEATER
    if ~condenser
        fun = @(Tin) precalentador(Tin, ph.Uph, ph.Aph, ph.Tv, ph.Tph_out, ph.Mf, ph.Cp);
%         x0  = ph.Tph_out-2; 
%         lb  = ph.Tph_out-5;
%         ub  = ph.Tph_out

        ph.Tph_in = fsolve(fun, ph.Tph_out-2, options_fsolve);

        ph.Mvh    = ( ph.Mf*ph.Cp*(ph.Tph_out-ph.Tph_in) ) / ...
                    ( enthalpySatVapTW(ph.Tv)-enthalpySatLiqTW(ph.Tv) );
        ph.Mv_out = ph.Mv_in - ph.Mvh;

        ph.Tvh    = ph.Tv;

    % CONDENSER
    else
        ph.Mvh = ph.Mv_in + ph.Mmix_in; % Total distillate produced (Mprod)
        
        % Obtain deltaPvc for the given conditions
%         ph.deltaPvc = ann_deltaPvc(ph.Mvh, ph.Tv, ph.Mf, ph.Tph_out);
        deltaPvc = 1;
        ph.Tv = ph.Tv * deltaPvc;
        ph.Tvh = ph.Tv; % Tprod, when no considering overcooling
        
        Tcwout = ph.Tph_out;
        
        ph.Mdf = ph.Mmix_in * ph.Cp_mix * (ph.Tmix_in-ph.Tv) / ...
                 ( enthalpySatVapTW(ph.Tv)-enthalpySatLiqTW(ph.Tv) );

        % At the moment it doesn't really implement the over cooling effect
        % so Mcw does nothing
        Tcwin = fsolve(@(Tcwin) condensador_with_over_cooling(Tcwin, ph.Uph, ph.Aph, ph.Tv, ...
                                             Tcwout, ph.Mv_in, ph.Mdf, ph.Mcw), ...
                                             25, options_fsolve);
        %% Debug
%         if ~isempty(ph_ref)
%             Tin = ph.Tph_out-10:0.1:ph.Tph_out; 
%             y = zeros(length(Tin), 3);
%             for idx=1:length(Tin)
%                 y(idx, 1) = condensador_with_over_cooling(Tin(idx), ph.Uph,     ph.Aph, ph.Tv, Tcwout, ph.Mv_in, ph.Mdf, ph.Mcw);
%                 y(idx, 2) = condensador_with_over_cooling(Tin(idx), ph_ref.Uph, ph.Aph, ph.Tv, Tcwout, ph.Mv_in, ph.Mdf, ph.Mcw);
%                 y(idx, 3) = condensador_with_over_cooling(Tin(idx), 10,         ph.Aph, ph.Tv, Tcwout, ph.Mv_in, ph.Mdf, ph.Mcw);
%             end
%             ax=nexttile;
%             plot(ax, Tin, y); hold on; 
%             yline(0, 'HandleVisibility','off');
%             scatter(ax, ph_ref.Tph_in, 0, 35, [0.4660 0.6740 0.1880], "filled",LineWidth=1.5)
%             scatter(ax, Tcwin, condensador_with_over_cooling(Tcwin, ph.Uph, ph.Aph, ph.Tv, Tcwout, ph.Mv_in, ph.Mdf, ph.Mcw), ...
%                     35, [0 0.4470 0.7410], Marker="x", LineWidth=1.5)
%                 legend(ax, sprintf('U=%.2f', ph.Uph), sprintf('ph_ref.Uph=%.2f', ph_ref.Uph), 'U=10','Interpreter', 'none')
%             title(ax, string)

%         end
%%
        ph.Tph_in = Tcwin;

        Mcw_no_over_cooling = ( ph.Mv_in +  ph.Mdf ) * ( enthalpySatVapTW(ph.Tv)-enthalpySatLiqTW(ph.Tv) ) ...
                                 / ( ph.Cp*(Tcwout-Tcwin) );
        if Mcw_no_over_cooling <0
            ph.Mcw_no_over_cooling = 0;
        else
            ph.Mcw_no_over_cooling = Mcw_no_over_cooling;
        end
%         if abs(Mf - Mprod - ef.Mb_out) > 1e-4 
%             warning("No se cumple balance de materia: Mf != Mb + Mprod (%.2f != %.2f)",...
%                     Mf*3600, (Mprod+ef.Mb_out)*3600)
%         end
    end

end