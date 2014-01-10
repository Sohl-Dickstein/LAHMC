% Author: Jascha Sohl-Dickstein, Mayur Mudigonda (2014)
% Web: http://redwood.berkeley.edu/mayur
% Web: http://redwood.berkeley.edu/jascha
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)

function [ E ] = E_rough( X, scale1, scale2 )
	cosX = cos(X*2*pi/scale2);
	E = sum( (X.^2) / (2*scale1^2) + cosX );
end

