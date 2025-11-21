function C = condensador_with_over_cooling(x, Uc, Ac, Tv, Tcwout, Mv_in, Mdf, Mcw)
% Función que devuelve temperatura de entrada del agua de refrigeración a
% la entrada de un condensador y temperatura de salida del condensado para
% un condensador en condiciones de subenfriamiento
%
% Salidas:
% x(1) = Tcwin
% x(2) = Tvh == Tprod



% if Tv-x < 0 || Tv-Tcwout < 0
%     C(1) = -9999; 
%     C(2) = 0;
% else
    C(1) = Ac*Uc*((Tv-x(1))-(Tv-Tcwout))/log(((Tv-x(1)))/(Tv-Tcwout)) - ...
           (Mv_in + Mdf)*(enthalpySatVapTW(Tv)-enthalpySatLiqTW(Tv));

%     C(2) = (Mv_in + Mdf)*enthalpySatVapTW(Tv) + (Mv_in+Mmix_in)*(Tvc-x(2)) - Mcw*Cp*(x(1)-Tcwout);
% end