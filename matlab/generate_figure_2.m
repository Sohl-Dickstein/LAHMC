% Compare Look Ahead Hamiltonian Monte Carlo (LAHMC) and standard Hamiltonian
% Monte Carlo (HMC) on 3 energy functions.  The output of this script produces
% Figure 2 and Table 1 in the paper:
%  Sohl-Dickstein, Jascha and Mudigonda, Mayur and DeWeese, Michael R.
%  Hamiltonian Monte Carlo Without Detailed Balance.
%  International Conference on Machine Learning. 2014
%
%
% Author: Jascha Sohl-Dickstein, Mayur Mudigonda (2014)
% Web: http://redwood.berkeley.edu/mayur
% Web: http://redwood.berkeley.edu/jascha
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)



clear all;
close all;

Nsamp = 500; % number of sampling steps to take
batch_size = 10;

%% uncomment the following 2 lines to reproduce exactly
%% the figures in the paper.  Runs much more slowly
%% than the lines above.
% Nsamp = 10000; % number of sampling steps to take
% batch_size = 400;

dt = datestr(now, 'yyyymmdd-HHMMSS');

% loop through the 3 models in the figure
for target_model_i = 1:3
    rng('default'); % make experiments repeatable

    fprintf( '\n\n\n' );
    
    opts_init = [];    
        
    % make this 1 for more output
    opts_init.Debug = 0;
    opts_init.M = 10;
    opts_init.epsilon = 1;
    opts_init.T = 1;
    opts_init.beta = 0.1;
    FEVAL_MAX = 5000000
    opts_init.funcevals = 0;
    
    if target_model_i == 1
        %% rough well
        DataSize = 2;
        
        modelname = sprintf('%dD_rough', DataSize);
        modeltitle = sprintf('%dD Rough', DataSize);
        
        opts_init.E = @E_rough;
        opts_init.dEdX = @dEdX_rough;
        theta = {100, 4};
              
        opts_init.Xinit = randn( DataSize, batch_size )*theta{1};
        %% burnin the samples
        tic();
        disp('burnin');
        opts_burnin = opts_init;
        opts_burnin.Debug = 1;
        opts_burnin.T = Nsamp*3;
        [Xloc, statesloc] = LAHMC( opts_burnin, [], theta{:});
        opts_init.Xinit = Xloc;
        toc()
        
        max_shift = 1001;
        
    elseif (target_model_i == 2) | (target_model_i == 3)
        %% anisotropic Gaussian
        
        if target_model_i == 3
            DataSize = 100;
        elseif target_model_i == 2
            DataSize = 2;
        end
        
        modelname = sprintf('%dD_Gaussian', DataSize);
        modeltitle = sprintf('%dD Anisotropic Gaussian', DataSize);
        
        opts_init.E = @E_gauss;
        opts_init.dEdX = @dEdX_gauss;
        theta = {diag(exp(linspace(log(1e-6), log(1), DataSize)))};
        opts_init.Xinit = sqrtm(inv(theta{1}))*randn( DataSize, batch_size );
        
        max_shift = 7001;
    end
    
    basedir = strcat(modelname, '_', dt, '/');
    basedirfig = strcat(basedir, 'figures', '/');
    mkdir(basedir);
    
    %Initalize Options
    ii = 1;
    names{ii} = 'HMC \beta=1';
    opts{ii} = opts_init;
    opts{ii}.mode = 'HMC';
    opts{ii}.beta = 1;
    %opts{ii}.alpha = 1
    %Initialize States
    states{ii} = [];
    % arrays to keep track of the samples
    X{ii} = zeros(DataSize,Nsamp);
    fevals{ii} = [];
    
    ii = ii + 1;
    names{ii} = 'LAHMC \beta=1';
    opts{ii} = opts_init;
    opts{ii}.mode = 'LAHMC';
    opts{ii}.beta = 1;
    %Initialize States
    states{ii} = [];
    % arrays to keep track of the samples
    X{ii} = zeros(DataSize,Nsamp);
    fevals{ii} = [];
    
    ii = ii + 1;
    names{ii} = 'HMC \beta=0.1';
    opts{ii} = opts_init;
    opts{ii}.mode = 'HMC';
    %Initialize States
    states{ii} = [];
    % arrays to keep track of the samples
    X{ii} = zeros(DataSize,Nsamp);
    fevals{ii} = [];
    
    ii = ii + 1;
    names{ii} = 'LAHMC \beta=0.1';
    opts{ii} = opts_init;
    opts{ii}.mode = 'LAHMC';
    %Initialize States
    states{ii} = [];
    % arrays to keep track of the samples
    X{ii} = zeros(DataSize,Nsamp);
    fevals{ii} = [];
    
    RUN_FLAG=1;
    ttt = tic();
    ii=1;
    % call the sampling algorithm Nsamp times
    while (ii <=Nsamp && RUN_FLAG == 1)
        for jj = 1:length(names)
            if ii == 1 || states{jj}.funcevals < FEVAL_MAX
                [Xloc, statesloc] = LAHMC( opts{jj}, states{jj},theta{:});
                states{jj} = statesloc;
                if ii > 1
                    X{jj} = cat(3,X{jj}, Xloc);
                else
                    X{jj} = Xloc;
                end
                
                fevals{jj}(ii,1) = states{jj}.funcevals;
                assert(batch_size == size(Xloc,2));
            else
                RUN_FLAG = 0;
                break;
            end
        end
        
        %Display + Saving
        if (mod( ii, ceil(Nsamp/50) ) == 0)
            fprintf('%d / %d in %f sec (%f sec remaining)\r', ii, Nsamp, toc(ttt), toc(ttt)*Nsamp/ii - toc(ttt) );
        end
        if (mod( ii, ceil(Nsamp/4) ) == 0) || (ii == Nsamp) || RUN_FLAG == 0
            fprintf('%d / %d in %f sec (%f sec remaining)\n', ii, Nsamp, toc(ttt), toc(ttt)*Nsamp/ii - toc(ttt) );
            
            for jj = 1:length(names)
                fprintf( '\n%s\n', names{jj});
                %disp(states{jj})
                %disp(states{jj}.steps)
                %disp(states{jj}.steps.leap')
                fprintf( 'total %f flip fraction %f L fraction: ', states{jj}.steps.total, states{jj}.steps.flip/states{jj}.steps.total );
                for kk = 1:length(states{jj}.steps.leap)
                    fprintf( '%f ', states{jj}.steps.leap(kk) / states{jj}.steps.total );
                end
                fprintf( '\n' );
                fprintf( 'Last sample L2 %f all sample L2 %f', mean(mean(X{jj}(:,:,end).^2)),  mean(mean(mean(X{jj}.^2))));
            end
            fprintf( '\n' );
            
            %Calculate average fevals by taking total fevals at this point
            %and dividing it by the number of samples we have acquired
            %fprintf('calculating average fevals');
            for jj=1:length(names)
                avg_fevals{jj}=fevals{jj}(end,1)/size(X{jj},3);
            end
            [h1,h2]=plot_autocorr_samples(X, names,avg_fevals, max_shift);
            %   disp('Autocorr plot completed')
            %             h2=plot_fevals(fevals, names);
            %          disp('Fevals plot completed')
            figure(h1);
            title(modeltitle);
            figure(h2);
            title(modeltitle);
            grid on
            drawnow;
            fp = strcat(basedir,'autocorr-fevals.fig');
            saveas(h1,fp);
            fp = strcat(basedir,'autocorr-steps.fig');
            saveas(h2,fp);
            save(strcat(basedir,'alldata.mat'));
            
            % use the export_fig util from https://sites.google.com/site/oliverwoodford/software/export_fig
            fp = strcat(basedir,'autocorr-fevals.pdf');
            try
                addpath('export_fig');
                export_fig( fp, h1 );
            catch err
                fprintf( '\nExpecting export_fig\n' );
            end
        end
        ii = ii + 1;
    end
    ttt = toc(ttt);

end
