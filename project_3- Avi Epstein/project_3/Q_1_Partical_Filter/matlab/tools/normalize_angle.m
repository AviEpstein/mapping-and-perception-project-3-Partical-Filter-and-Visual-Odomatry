function [phiNorm] = normalize_angle(phi)
%Normalize phi to be between -pi and pi

if (phi>pi)
    phi = phi - 2*pi;
else
    
    if (phi<-pi)
        phi = phi + 2*pi;
    end
end
phiNorm = phi;

end
