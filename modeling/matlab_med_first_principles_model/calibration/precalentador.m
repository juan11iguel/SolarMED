function PH = precalentador(Tin, U, A, Tv, Tout, Mf, Cp)
%     if Tv-Tin < 0 || Tv-Tout < 0
%         PH = 999;
% %         warning("Tin o Tout en precalentador > Tv")
%     else
        PH = Mf*Cp*(Tout-Tin) * log((Tv-Tin)/(Tv-Tout)) - (U*A*((Tv-Tin)-(Tv-Tout)));
%     end
end
            