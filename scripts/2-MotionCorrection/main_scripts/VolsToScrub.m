function vol_toscr = VolsToScrub(vol_i,nv,lower_thresh,higher_thresh)

vol_toscr = [];
for i =1:length(vol_i)
 
    if vol_i(i) > 1 && vol_i(i) <= nv-2
        for j = lower_thresh:higher_thresh
            vol_toscr = [vol_toscr,vol_i(i)+j];
        end
    elseif vol_i(i) == 1
        for j = 0:higher_thresh
            vol_toscr = [vol_toscr,vol_i(i) + j];
        end
    elseif vol_i(i) == nv-1
        for j = lower_thresh:1
            vol_toscr = [vol_toscr,vol_i(i) + j];
        end
    elseif vol_i(i) == nv
        for j = lower_thresh:0
            vol_toscr = [vol_toscr,vol_i(i) + j];
        end
    end
 
end
vol_toscr = unique(vol_toscr);

