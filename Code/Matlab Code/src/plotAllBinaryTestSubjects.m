        %%% plot which faces its comparin'
        
        % class 1
        figure
        subplot(2, 2, 1)
        imagesc(reshape(faceMatrix(:, ...
                               numImagesPerClass*i-(numImagesPerClass-1)), ...
                    nRow, nCol));
        colormap gray;
        title('1st image from class 1')
        axis equal
        
        subplot(2, 2, 2)
        imagesc(reshape(faceMatrix(:, numImagesPerClass*i), ...
                    nRow, nCol));
        colormap gray;
        title('2nd image from class 1')
        axis equal
        
        % class 2
        subplot(2, 2, 3)
        imagesc(reshape(faceMatrix(:, ...
                               numImagesPerClass*j-(numImagesPerClass-1)), ...
                    nRow, nCol));
        colormap gray;
        title('1st image from class 2')
        axis equal
        
        subplot(2, 2, 4)
        imagesc(reshape(faceMatrix(:, numImagesPerClass*j), ...
                    nRow, nCol));
        colormap gray;
        title('2nd image from class 2')
        axis equal
        

