function y = calculoUef(U,Tv_des,i,ef)
    options = optimoptions(@fsolve,'Display','none');

    if i == 1
        if strcmp(ef(i).source, 'steam')
            x0 = [ef.Ts_in-5,ef.Mb_in*0.1,ef.Mb_in*0.9,ef.Xb_in*1.1,ef.Ts_in-5];
            a = fsolve(@(x) efecto_1(x, U, ef.Aef, ef.Ms, ef.Ts_in, ef.Tb_in, ef.Mb_in, ef.Xb_in),x0,options);
            TV = a(5); 
        elseif strcmp(ef(i).source, 'liquid')
            x0 = [ef.Ts_in-5,ef.Mb_in*0.1,ef.Mb_in*0.9,ef.Xb_in*1.1,ef.Ts_in-5];
            a = fsolve(@(x) efecto_1_LIQ(x, U, ef.Aef, ef.Ms, ef.Ts_in, ef.Ts_out, ef.Tb_in, ef.Mb_in, ef.Xb_in, ef.Cp_s),x0,options);
            TV = a(5); 
        else
            warning("No se ha especificado estado de fuente de calor (steam or liquid)")
        end
    else 
        
        x0 = [ef.Mb_in*0.1,ef.Mb_in,ef.Xb_in,ef.Tv_in-2.7,ef.Tb_in-2.7,0,...
            ef.Mb_in,ef.Xb_in,ef.Tb_in-2.7,ef.Mv_in, ef.Mdest, 0, ef.Tdest, ef.Tdest-2.7];
    
        a = fsolve(@(x) efecto_i_flash(x,ef,U), x0, options);
        TV = a(4); 
    end
 y = (TV - Tv_des)^2;