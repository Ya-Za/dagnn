function tile(rows, cols)
    % rows, cols
    if nargin < 2
        n = length(findobj('Type', 'Figure'));
        rows = ceil(sqrt(n));
        cols = ceil(n / rows);
    end
    
    tileFigures(rows, cols);
    createOverallFigure(rows, cols);
end

function tileFigures(rows, cols)
    % screen size
    root = findobj('Type', 'Root');
    screenSize = root.ScreenSize;
    width = screenSize(3);
    height = screenSize(4);
    
    % figures
    root.Children = root.Children(end:-1:1);
    figs = findobj('Type', 'Figure');
    
    % tile
    W = ceil(width / cols);
    H = ceil(height / rows);
    
    index = 1;
    n = length(figs);
    
    for r = rows:-1:1
        for c = 1:cols
            x = (c - 1) * W;
            y = (r - 1) * H;
            figs(index).Position = [x, y, W, H];
            
            index = index + 1;
            
            if index > n
                return;
            end
        end
    end
end

function createOverallFigure(rows, cols)
    % axes
    axs = findobj('Type', 'Axes');
    
    % overall figure
    fig = figure(...
        'Name', 'Overall', ...
        'Color', 'white', ...
        'NumberTitle', 'off', ...
        'Units', 'normalized', ...
        'OuterPosition', [0, 0, 1, 1] ...
    );

    % tile
    for i = 1:length(axs)
        ax = copyobj(axs(i), fig);
        subplot(rows, cols, i, ax);
    end
end