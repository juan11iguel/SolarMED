function C = condensador(Tcw_in, Uc, Ac, Tv, Tcw_out, Mgb, Mgf, Mdf)
if Tv-Tcw_in < 0 || Tv-Tcw_out < 0
    C = 0;
else
    C = Ac*Uc*((Tv-Tcw_in)-(Tv-Tcw_out))/log(((Tv-Tcw_in))/(Tv-Tcw_out))...
    - (Mgb + Mgf + Mdf)*(enthalpySatVapTW(Tv)-enthalpySatLiqTW(Tv));
end