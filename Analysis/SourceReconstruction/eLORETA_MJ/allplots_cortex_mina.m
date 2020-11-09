function allplots_cortex_mina2(sa, data, colorlimits, cm, unit, smooth, varargin)
%% varargin options
% example: (...,'views', 1:8, 'save', 1, 'namesave','mysources','saveformat','epsc')

% views -----------
% 1 = left lateral, 2 = left hem medial
% 3 = rightlateral, 4 = right medial
% 5 = dorsal, 6 = dorsal horizontal
% 7 = ventral, 8 = ventral horizontal
% ----------

% save : if = 1,  figures will be saved, default = 0
% namesave : the name of the figures. default='source'
% saveformat: format of the saved figure. without dot. default = 'epsc'
 
%% Notes
% The function is taken from Stefan. It has been modified by \Mina. 
% - varargin1 is deleted. 
% - print directory is omited
% - varargin handling is based on Mina's preferences:D
% - plots are rearranged
%%
set(0,'DefaultFigureColor',[1 1 1])
if smooth 
    vc = sa.cortex75K.vc_smooth;
    sm = '_smooth';
else
    vc = sa.cortex75K.vc;
    sm = '';
end
surface_pars = struct('alpha_const', 1, 'colormap', cm, 'colorlimits', colorlimits, ...
  'showdirections', 0, 'colorbars', 0, 'dipnames', [], 'mymarkersize', 15, 'directions', [0 0 1 1 1 1], ...
  'printcbar', 1, 'userticks', []);
%% handel varargin - added by \Mina

%defaults
DoSave = 0;
saveformat = 'epsc';
savenameflag = 0;
saveName = 'source';

% check varargin
if (rem(length(varargin),2) == 1)
    error('Optional parameters should always go by pairs');
else
    for j = 1:2:(length(varargin)-1)
        if ~ischar (varargin{j})
            error (['Unknown type of optional parameter name (parameter' ...
                ' names must be strings).']);
        end
        switch lower (varargin{j})
            case 'views'
                views = varargin{j+1};
            case 'save'
                DoSave = varargin{j+1};
            case 'savename'
                savenameflag = 1;
                saveName = varargin{j+1};
            case 'saveformat'
                saveformat = varargin{j+1};
        end
        
    end
end

if DoSave
    if ~savenameflag
        warning('No name for figures specified. The default name ((source)) is used')
    end
end
%% plots - rearranged by \Mina
% left hemisphere ------------------
surface_pars.myviewdir = [-1 0 0]; 

%  view =1, left lateral
if ismember(1,views)    
    figure; showsurface3(vc, sa.cortex75K.tri_left, surface_pars, data);
    if DoSave
        saveas(gcf,[saveName,'_left_lat'],saveformat);
    end
end

% view = 2; left medial
if ismember(2,views)
    figure; showsurface3(vc, sa.cortex75K.tri_right, surface_pars, data);
    if DoSave
        saveas(gcf,[saveName,'_left_med'],saveformat);
    end
end


% right hemisphere ------------------
surface_pars.myviewdir = [1 0 0];

%  view = 3, right lateral
if ismember(3,views)    
    figure; showsurface3(vc, sa.cortex75K.tri_right, surface_pars, data);
    if DoSave
        saveas(gcf,[saveName,'_right_lat'],saveformat);
    end
end

% view = 4; right medial
if ismember(4,views)
    figure; showsurface3(vc, sa.cortex75K.tri_left, surface_pars, data);
    if DoSave
        saveas(gcf,[saveName,'_right_med'],saveformat);
    end
end


% Dorsal View ------------------
surface_pars.myviewdir = [0 0 1];

%  view = 5, dorsal view
if ismember(5,views)    
    figure; showsurface3(vc, sa.cortex75K.tri, surface_pars, data);
    if DoSave
        saveas(gcf,[saveName,'_dorsal'],saveformat);
    end
end

% view = 6; dorsal view - horizontal
if ismember(6,views)
    surface_pars.myviewdir = [-1e-10 0 1];
    surface_pars.directions = [1 0 1 1 0 0];
    figure; showsurface3(vc, sa.cortex75K.tri, surface_pars, data);%upperview rotated
    if DoSave
        saveas(gcf,[saveName,'_dorsal_horiz'],saveformat);
    end
end


% Ventral View ------------------
surface_pars.myviewdir = [-1e-10 0 -1]; 

%  view = 7, ventral view
if ismember(7,views)    
    figure; showsurface3(vc, sa.cortex75K.tri, surface_pars, data);
    if DoSave
        saveas(gcf,[saveName,'_vent'],saveformat);
    end
end

% view = 8; ventral view - horizontal
if ismember(8,views)
    surface_pars.myviewdir = [0 1e-10 -1];
    figure; showsurface3(vc, sa.cortex75K.tri, surface_pars, data); 
    
    if DoSave
        saveas(gcf,[saveName,'_vent_horiz'],saveformat);
    end
end

% color bar ------------------
if isfield(surface_pars, 'printcbar') && surface_pars.printcbar
  figure; 
  hf = imagesc(randn(5)); colormap(cm)
  set(gca, 'clim', colorlimits, 'position', [0.1 0.1 0.6 0.8], 'visible', 'off')
  set(hf, 'visible', 'off')
  cb = colorbar; 
  set(cb, 'fontsize', 30)
  if ~isempty(surface_pars.userticks)
    set(cb, 'xtick', sort([colorlimits, surface_pars.userticks]))
  end
  ylabel(cb, unit)
end
if DoSave
  saveas(gcf,[saveName,'_scale'],saveformat);
end

set(0,'DefaultFigureColor',[1 1 1])





