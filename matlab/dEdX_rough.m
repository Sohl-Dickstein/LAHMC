% Author: Jascha Sohl-Dickstein, Mayur Mudigonda (2014)
% Web: http://redwood.berkeley.edu/mayur
% Web: http://redwood.berkeley.edu/jascha
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)

function [ dEdX ] = dEdX_rough( X, scale1, scale2 )
	sinX = sin(X*2*pi/scale2);
	dEdX = X./scale1^2 + -sinX*2*pi./scale2;
end

