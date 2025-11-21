function y = calculoUph_cls(U,Tph_des,i,Nef,ph)
    options = optimoptions('fsolve','Display','none');
    
    if i == Nef
        TPH = fsolve(@(Tcw_in) condensador(...
              Tcw_in, U, ph.Aph, ph.Tv, ph.Tph_out, ph.Mgb, ph.Mgf, ph.Mdf),...
              ph.Tph_out-1, options);
    else
        TPH = fsolve(@(Tin) precalentador(...
              Tin, U, ph.Aph, ph.Tv, ph.Tph_out, ph.Mf, ph.Cp), ...
              ph.Tph_out-5, options);
    end
%     TPH_IN = TPH + Tph_des;
%     fprintf('PH %d. Error = %.2f, T=%.2f, U=%.2f\n',i, (TPH - Tph_des)^2, TPH, U);
    
y = abs(TPH - Tph_des);