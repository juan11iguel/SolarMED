function F1 = efecto_1(x, Uef, Aef, Ms, Ts_in, Tf, Mf, Xf)
    % Tb1 = x(1)
    % Mgb1 = x(2)
    % Mb1 = x(3)
    % Xb1 = x(4) 
    % Tv1 = x(5)   
    
    cp = SW_SpcHeat(x(1),x(4))/1000;
    
    F1(1) = Ts_in - (Ms*(enthalpySatVapTW(Ts_in)-enthalpySatLiqTW(Ts_in))) / (Uef*Aef) - x(1);
    F1(2) = (Ms*(enthalpySatVapTW(Ts_in)-enthalpySatLiqTW(Ts_in)) - Mf*cp*(x(1)-Tf)) / (enthalpySatVapTW(x(5))-enthalpySatLiqTW(x(5))) - x(2);
    F1(3) = Mf - x(2) - x(3);
    F1(4) = Mf/x(3)*Xf - x(4);
    F1(5) = x(1) - SW_BPE(x(1),x(4)) - x(5);
end