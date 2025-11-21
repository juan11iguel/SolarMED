function Fi = efecto_i_flash(x, ef, Uef)
    % Mgbi = x(1)
    % Mbi  = x(2)
    % Xbi  = x(3) 
    % Tvi  = x(4)
    % Tbb  = x(5)  (Tb')
    % Mgfi = x(6)
    % Mbb  = x(7)  (Mb')
    % Xbb  = x(8)  (Xb')
    % Tbi  = x(9)
    % Mv_in     = x(10)
    % Mdest_in  = x(11)
    % Mdest_f   = x(12)
    % Tdest_in  = x(13)
    % Tdest_out = x(14)
    
    Mv_ant = ef.Mv_in;
    Tv_ant = ef.Tv_in;
    
    Mb_ant = ef.Mb_in;
    Tb_ant = ef.Tb_in;
    Xb_ant = ef.Xb_in;
    
    Mdest = ef.Mdest;
    Tdest = ef.Tdest;
    
    Aef = ef.Aef;

    deltaHTvi    = enthalpySatVapTW(x(4))   - enthalpySatLiqTW(x(4));
    deltaHTviant = enthalpySatVapTW(Tv_ant) - enthalpySatLiqTW(Tv_ant);
    deltaHdest   = enthalpySatLiqTW(x(13))  - enthalpySatLiqTW(x(4));
    
    cp_b  = SW_SpcHeat(x(9),x(3))/1000;
    cp_dest = cpSatLiqTW(Tdest);
    
    aporte_vapor_ant = x(10)*(deltaHTviant);   % aporte_vap = Mv_in * deltaHTv_ant
    aporte_dest = x(11)*cp_dest*(x(13)-x(14)); % aporte_dest = Mdest_in * Cp * (Tdest_in-Tdest_out)
    aporte_brine_liq = x(7)*cp_b*(x(9)-x(5));  % aporte_brine = Mb' * Cp * (Tb - Tb')
    
    Fi(1) =  (aporte_vapor_ant + aporte_dest - aporte_brine_liq)/(deltaHTvi) - x(1);
    Fi(2) = x(7) - x(1) - x(2); % Mb = Mbb - Mgb
    Fi(3) = x(7)*x(8)/x(2) - x(3);  % Xb = Mb'/Mb * Xb'
    Fi(4) = x(9) - SW_BPE(x(9),x(3)) - x(4);     % Tv  = Tb - BPE
    Fi(5) = x(9) + NEA(x(9),Tb_ant,x(4)) - x(5); % Tb' = Tb + NEA
    Fi(6) = Mb_ant*cp_b*(Tb_ant-x(5))/(deltaHTvi) - x(6); % Mgf = Mb_ant*Cp*(Tb_ant-Tb')/deltaHTv
    Fi(7) = Mb_ant - x(6) - x(7); % Mb' = Mb_ant - Mgf
    Fi(8) = (Xb_ant * Mb_ant) / x(7) - x(8);    % Xb' = Xb_ant * Mb_ant/Mb'
    Fi(9) = Tv_ant - (Mv_ant*(deltaHTviant) / (Uef*Aef)) - x(9); % Tb = Tv_ant - Mv_ant*deltaHTvant/UA
    Fi(10) = Mv_ant + x(12) - x(10);    % Mv_in = Mv_ant + Mdest_f
    Fi(11) = Mdest - x(12) - x(11);     % Mdest_in = Mdest - Mdest_f
    Fi(12) = x(11)*cp_dest*(Tdest - x(13))/deltaHTvi - x(12); % Mdest_f = Mdest*Cp*(Tdest-Tdest')/deltaHTv
    Fi(13) = x(4) + NEA(x(13), Tdest, x(4)) - x(13);    % Tdest_in = Tv + NEA
    Fi(14) = x(13) - deltaHdest/cpSatLiqTW(x(13)) - x(14); % Tdest_out = Tdest_in - AHdest/Cp
    
end
