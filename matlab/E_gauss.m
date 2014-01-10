% Author: Jascha Sohl-Dickstein, Mayur Mudigonda (2014)
% Web: http://redwood.berkeley.edu/mayur
% Web: http://redwood.berkeley.edu/jascha
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)

function E = E_gauss( X, J )
	% return the energy for each sample (column vector) in X for a Gaussian with
	% inverse covariance matrix (coupling matrix) J        
    E = 0.5*sum( X.*(J*X), 1 );
end