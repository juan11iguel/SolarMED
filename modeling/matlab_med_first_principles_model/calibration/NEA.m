function y = NEA(Tb, Tb_ant, Tv)
% Falta introducir referencia de dónde se haya sacado la expresión
    if Tb_ant-Tb < 0
        y = 0;
    else
        y = 33*(Tb_ant-Tb)^0.55/Tv;
    end
end
