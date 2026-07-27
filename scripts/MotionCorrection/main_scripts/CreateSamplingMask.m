function Mask = CreateSamplingMask(option,ilacq,sms_fac,n1,n2,nsl,nv,index)

l = length(ilacq);
na = n1*n2;


n12 = n1*n2;

switch option
    case 'fulldata'
        nb = nsl*nv/sms_fac;
        Mask = zeros(na,nb);

        k1=1;
        count=0;
        for i1 = 1:nv
            for j1 = 1:nsl
                Mask((ilacq(j1)-1)*n12 + 1: ilacq(j1)*n12,k1)  = 1;
                if count >= sms_fac-1
                    k1 = k1+1;
                    count=-1;
                end
                count=count+1;
            end
        end
        
    case 'partialdata'
         
        nb = nsl*nv/sms_fac;
        Mask = zeros(na,nb);

        k1=1;
        count_sl = 0;
        for i1 = 1:nv
            count=1;
            for j1 = 1:l % l-  reflects fewer slices
                Mask((ilacq(j1)-1)*n12 + 1: ilacq(j1)*n12,k1)  = 1;
                if ismember(count,index) || count_sl>=sms_fac-1
                    k1=k1+1;
                    count_sl = -1;
                end
                count_sl = count_sl+1;
                count=count+1;
            end
        end
        
    otherwise 
        warning('No correct option entered and no mask created.')
        
end