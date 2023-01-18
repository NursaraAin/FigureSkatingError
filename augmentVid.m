path = fullfile("C:\Users\nursa\OneDrive - Universiti Malaya\FigureSkatingError\under-rotation\full rotation");
fds = fileDatastore(path,"ReadFcn",@load,'FileExtensions','.mp4');
n = numel(fds.Files);
%%
for i = 1:n
    vidPath=split(fds.Files(i),"\");
    vidName=string(vidPath(numel(vidPath)));
    vid=VideoReader(vidName);
    newVid=sprintf("f-%d.mp4",i);
    createV=VideoWriter(newVid,'MPEG-4');
    open(createV);
    while hasFrame(vid)
        frame = readFrame(vid);
        %reflect=flip(frame,2);
        writeVideo(createV,frame);
    end
    close(createV);
end