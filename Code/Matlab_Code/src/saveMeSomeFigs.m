function saveMeSomeFigs(shouldSaveFigures, saveTitle)
    
    %%% saveMeSomeFigs(shouldSaveFigures, saveTitle)
    %%%
    %%% Takes a boolean toggle (shouldSaveFigures) and a string with the
    %%% save name (saveTitle), including the path, that you want to save
    %%% the current figure as. Example saveTitle looks like the following:
    %%%
    %%%     saveTitle = '../Figures/Velocity vs Time.pdf' - this would save
    %%%     a figure named 'Velocity vs Time.pdf' (as a .pdf) to a folder
    %%%     called 'Figures' up one directoy from the code's working
    %%%     directory.
    %%% 
    %%% Uses the gca function to pull the current figure, normalize
    %%% and scale it to the deafult paper size, and save it as a .pdf.
    %%%
    %%%
    %%% Examples function call:
    %%%
    %%% x = linspace(0, 2*pi, 100);
    %%% y = sin(x);
    %%% plot(x, y)
    %%% title('saveMeSomeFigs Test')
    %%%
    %%% saveTitle = 'saveMeSomeFigs Test Plot.pdf';
    %%% shouldSaveFigures = true;
    %%% saveMeSomeFigs(shouldSaveFigures, saveTitle)
    %%%
    %%% Last Modified: 5/3/2017
    %%% Date Created: 2/10/2017
    %%% Author: Nicholas Renninger

    % setup and save figure as .pdf
    if shouldSaveFigures
        curr_fig = gcf;
        set(curr_fig, 'PaperOrientation', 'landscape');
        set(curr_fig, 'PaperUnits', 'normalized');
        set(curr_fig, 'PaperPosition', [0 0 1 1]);
        [fid, errmsg] = fopen(saveTitle, 'w+');
        
        if fid < 1 % check if file is already open.
            error('Error Opening File in fopen: \n%s', errmsg); 
        end
        
        fclose(fid);
        print(gcf, '-dpdf', saveTitle);
    end

end