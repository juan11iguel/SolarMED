function F1 = efecto_1_LIQ(x, Uef, Aef, Ms, Ts_in, Ts_out, Tf, Mf, Xf, cp_s)
    % Tb1 = x(1)
    % Mgb1 = x(2)
    % Mb1 = x(3)
    % Xb1 = x(4) 
    % Tv1 = x(5)   
    
    cp = SW_SpcHeat(x(1),x(4))/1000;

%     F1(1) = Ts_in - (Ms*(Ts_in-Ts_out)*Cp) / (Uef*Aef) - x(1);
    if Ts_in < Tf || Ts_out < x(1)
        F1(1) = 0;
        warning('modelo_efecto1_LIQ:nonFeasibility',"Ts_in (%.1f) < Tf (%.1f) o Ts_out (%.1f) < Tb (%.1f)",Ts_in, Tf, Ts_out, x(1));
    else
        F1(1) = Ms*(Ts_in-Ts_out)*cp_s * log( (Ts_in-Tf)/(Ts_out-x(1)) ) - Uef*Aef*( (Ts_in-Tf)-(Ts_out-x(1)) );
    end
    
    F1(2) = (Ms*(Ts_in-Ts_out)*cp_s - Mf*cp*(x(1)-Tf)) / (enthalpySatVapTW(x(5))-enthalpySatLiqTW(x(5))) - x(2);
    F1(3) = Mf - x(2) - x(3);
    F1(4) = Mf/x(3)*Xf - x(4);
    F1(5) = x(1) - SW_BPE(x(1),x(4)) - x(5);
end