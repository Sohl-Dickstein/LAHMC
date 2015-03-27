% Implements Look Ahead Hamiltonian Monte Carlo (LAHMC) and standard 
% Hamiltonian Monte Carlo (HMC).  See the associated paper:
%   Sohl-Dickstein, Jascha and Mudigonda, Mayur and DeWeese, Michael R.
%   Hamiltonian Monte Carlo Without Detailed Balance.
%   International Conference on Machine Learning. 2014
%
% [X, state] = LAHMC(opts, state, [additional arguments])
%
% Inputs:
%   opts - A structure setting sampling options.  Fields are:
%      Xinit - The initial particle positions for sampling.  Should be an
%        array of size [number dimensions]x[batch size].  (required field)
%      E - Function that returns the energy.  E(X, varargin{:}) should return
%        a 1x[batch size] array, with each entry corresponding to the energy
%        of the corresponding column of X.  (required field)
%      dEdX - Function that returns the gradient.  dEdX(X, varargin{:}) should
%        return an array of the same size as X,
%        [number dimensions]x[batch size].  (required field)
%      T - Number of sampling steps to take. (default 100)
%      epsilon - Step length. (default 0.1)
%      M - The number of Leapfrog steps per simulation run. (default 10)
%      K - The maximum number of times to apply the L operator. (default 4)
%      alpha - One way to specify momentum corruption rate.  See Eq. 22.
%        (default 0.2)
%      beta - Alternate way to specify momentum corruption rate.  See Eq. 15.
%        (default value derived from alpha)
%      mode - 'LAHMC' or 'HMC'. (default 'LAHMC')
%   state - The internal state of the sampler.  Set to [] the first time LAHMC
%     is called.  Pass the returned state back in to continue a sampling chain
%     across multiple calls to LAHMC.
%   [additional arguments] - Optional.  Any additional arguments are passed
%     on to the E and dEdX functions.
%     
% Outputs:
%   X - The most recent sample for every particle in the batch.
%   state - Holds the complete state of the sampler, and can be passed back in
%     to resume sampling where the last function call to LAHMC left off.
%
% See README.md for example code calling LAHMC.
%
% Author: Mayur Mudigonda, Jascha Sohl-Dickstein (2014)
% Web: http://redwood.berkeley.edu/mayur
% Web: http://redwood.berkeley.edu/jascha
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)


function [X, state] = LAHMC( opts, state, varargin )   
    % load parameters
    f_E = getfield(opts, 'E');
    f_dEdX = getfield(opts, 'dEdX');
    T = getField( opts, 'T', 100 );
    epsilon = getField( opts, 'epsilon', 0.1 );
    M = getField(opts, 'M', 10);
    K = getField( opts, 'K', 4);
    alpha = getField( opts, 'alpha', 0.2);
    beta = alpha^(1 / (epsilon*M));
    beta = getField( opts, 'beta', beta);    
    sampler_mode = getField( opts, 'mode', 'LAHMC');
    assert( strcmp(sampler_mode, 'HMC') | strcmp(sampler_mode, 'LAHMC') );
    
    debug_display = getField( opts, 'debug_display', 0 );

    % opts gets passed into other functions -- make these available there too
    opts.M = M; 
    opts.epsilon = epsilon;

    % initialize the state variable if not already initialized
    if isempty(state)
        state.X = getfield(opts, 'Xinit');
        szb = size(state.X, 2);
        szd = size(state.X, 1);

        state.V = randn( szd, szb );
        % state.steps provides counters for each kind of transition
        state.steps = [];
        state.steps.leap = zeros(K,1);
        state.steps.flip = 0;
        state.steps.stay = 0;
        state.steps.total = 0;
        state.funcevals = 0;
        % populate the energy and gradient at the initial position       
        state.dEdX = f_dEdX( state.X, varargin{:} );
        state.E = f_E( state.X, varargin{:} );
    end
    szb = size(state.X, 2);
    szd = size(state.X, 1);

    % used to accumulate gradient evaluations for plots in paper
    global funcevals_inc;
    funcevals_inc = 0;
    
    for t = 1:T % iterate over update steps
        if debug_display
            fprintf( '.' );
        end
        
        L_state = leap_HMC(state,[],opts,varargin{:});
        r_L = leap_prob(state,L_state,[]);
        % compare against a random number to decide whether to accept
        rnd_cmp = rand(1,szb);
        gd = (rnd_cmp < r_L);
        % update the current state for the samples where the forward transition
        % was accepted
        if sum(gd) > 0
            state = update_state(state,L_state,gd);
            state.steps.leap(1) = state.steps.leap(1) + sum(gd);
        end
        % bd indexes the samples for which the forward transition was rejected
        bd = (rnd_cmp >= r_L);
        %If there are samples that are rejected
        if sum(bd) > 0
            switch sampler_mode
                %Standard HMC flipping = 1 - leap
                case 'HMC'
                    state = flip_HMC(state,bd);
                    state.steps.flip = state.steps.flip + sum(bd);
                case 'LAHMC'
                    % n steps
                    state_ladder = {};
                    bd_idx = find(bd);
                    %bd_lad
                    state_ladder{1} = state; % Present State
                    state_ladder{2} = L_state; % Leap State

                    % this matrix will hold the cumulative probability of transitioning between pairs of states
                    C = ones(K+1,K+1,szb)*nan;

                    % Greedily evaluate how far each particle should leap, up to L^K steps
                    for nn = 3:K+1
                        state_ladder{nn} = leap_HMC(state_ladder{nn-1}, bd_idx, opts, varargin{:});
                        [~,p_cum,Cl] = leap_prob_recurse(state_ladder(1:nn), C(1:nn,1:nn,bd_idx), bd_idx);
                        C(1:nn,1:nn,bd_idx) = Cl;
                        jump_bin = rnd_cmp(bd_idx) < p_cum;
                        jump_idx = bd_idx(jump_bin);
                        state = update_state(state,state_ladder{nn},jump_idx);
                        bd_idx = bd_idx(~jump_bin);
                        state.steps.leap(nn-1) = state.steps.leap(nn-1) + length(jump_idx);
                        if length(bd_idx) == 0
                            % all the particles in the batch have found a transition
                            break
                        end
                    end
                    % and if there are any left, flip them
                    if length(bd_idx) > 0
                        state = flip_HMC(state,bd_idx);
                        state.steps.flip = state.steps.flip + length(bd_idx);
                    end
            end
        end

        % corrupt the momentum
        N1 = randn( szd, szb );      
        state.V  = real(sqrt(1-beta)) * state.V + sqrt(beta) * N1;
        state.steps.total = state.steps.total + szb;    
    end
    
    state.funcevals = state.funcevals + funcevals_inc/szb;
    
    X = state.X;
end

function [v] = getField(options,opt,default)
    % to process the fields in our options structure
    % this function taken from Mark Schmidt's minFunc
    % http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html
    options = toUpper(options); % make fields case insensitive
    opt = upper(opt);
    if isfield(options,opt)
        if ~isempty(getfield(options,opt))
            v = getfield(options,opt); 
        else
            v = default;
        end
    else
        v = default;
    end
end
function [o] = toUpper(o)
    if ~isempty(o)
        fn = fieldnames(o);
        for i = 1:length(fn)
            o = setfield(o,upper(fn{i}),getfield(o,fn{i}));
        end
    end
end

function orig_state = update_state(orig_state,update_state,indices)
    % Copy the particles indicated by 'indices' from 'update_state' to 'orig_state'

    orig_state.X(:,indices) = update_state.X(:,indices);
    orig_state.V(:,indices) = update_state.V(:,indices);
    orig_state.E(:,indices) = update_state.E(:,indices);
    orig_state.dEdX(:,indices) = update_state.dEdX(:,indices);
end

function [state] = flip_HMC(state,ind)
    % apply the momentum flip operator to the particles identified by 'ind' in 'state'

    if nargin < 2
        ind = 1:size(state.V,2);
    end
    state.V(:,ind) = -state.V(:,ind);
end

function [state] = leap_HMC(state,ind,opts,varargin)
    % apply leap operator to the particles identified by 'ind' in 'state'

    global funcevals_inc

    if isempty(ind)
        ind = true(size(state.V,2),1);
    end
    Nind = sum(ind);

    f_E = opts.E;
    f_dEdX = opts.dEdX;
    % dut = 0.5; % fraction of the momentum to be replaced per unit time
    epsilon = opts.epsilon;
    M = opts.M;

    % run M leapfrog steps
    V = state.V(:,ind);
    X = state.X(:,ind);
    dEdX = state.dEdX(:,ind);

    for ii=1:M
        funcevals_inc = funcevals_inc + Nind;
        
        V0 = V;
        X0 = X;
        dEdX0 = dEdX;
        Vhalf = V0 - epsilon/2 * dEdX0;
        X1 = X0 + epsilon * Vhalf;
        dEdX1 = f_dEdX( X1, varargin{:} );
        V1 = Vhalf - epsilon/2 * dEdX1;

        V = V1;
        X = X1;
        dEdX = dEdX1;
    end
    state.X(:,ind) = X;
    state.V(:,ind) = V;
    state.dEdX(:,ind) = dEdX;
    state.E(:,ind) = f_E( X, varargin{:});
end

function [H] = hamiltonian_HMC(state,ind)
    % function to evaluate Hamiltonian of a state
    if isempty(ind)
        ind = 1:size(state.V,2);
    end
    E = state.E(:,ind);
    V = state.V(:,ind);

    % TODO store the result of (1/2) * (sum(V.*V)) so we don't have to constantly recalculate it
    H = E + (1/2) * (sum(V.*V));
end

function [resid, cumu, C] = leap_prob_recurse(state_ladder, C, idx)
    % returns [residual probability], [cumulative probability of transition from first to last state on ladder], [matrix of cumulative probabilities between all pairs of states]

    if isfinite(C(1,end,1))
        %fprintf('a')

        % this has already been calculated
        resid = 1 - C(1,end,:);
        cumu = C(1,end,:);
        resid = resid(:)';
        cumu = cumu(:)';
        return
    end

    if size(state_ladder,2) == 2
        % the two states are one apart
        prob = leap_prob(state_ladder{1}, state_ladder{2}, idx);
        cumu = prob;
        resid = 1 - prob;
        C(1,end,:) = cumu(:);
        return;
    end
       
    [residual_forward, cumulative_forward, Cl] = leap_prob_recurse(state_ladder(1:end-1), C(1:end-1,1:end-1,:), idx);
    C(1:end-1,1:end-1,:) = Cl;
    [residual_reverse, cumulative_reverse, Cl] = leap_prob_recurse(state_ladder(end:-1:2), C(end:-1:2,end:-1:2,:), idx);
    C(end:-1:2,end:-1:2,:) = Cl;

    H1 = hamiltonian_HMC(state_ladder{1},idx);
    H2 = hamiltonian_HMC(state_ladder{end}, idx);
    start_state_ratio = exp(H1 - H2);
    prob = min([residual_forward; residual_reverse.*start_state_ratio], [], 1);
    cumu = cumulative_forward + prob;
    resid = 1 - cumu;
    C(1,end,:) = cumu(:);
end

% probability of accepting a leap transition under standard
% Metropolis-Hastings
function [prob] = leap_prob(start_state, leap_state,idx)
    H_leap = hamiltonian_HMC(leap_state,idx);
    H_start = hamiltonian_HMC(start_state,idx);
    prob = min(1,exp(H_start - H_leap));
end
