close('all');
clear;
clc;

%--------------------
%Downloads DAGNN
%--------------------
disp('Downloading DAGNN... Please be patient');
urlwrite('https://github.com/Ya-Za/dagnn/archive/master.zip', 'dagnn.zip');
disp('Downloaded! Unziping');
unzip('dagnn.zip');
movefile('dagnn-master', 'dagnn');
delete('dagnn.zip');
cd('dagnn');

%--------------------
%Checks for Toolboxes
%--------------------

required_toolboxes = [{'Signal_Toolbox','Signal Processing Toolbox'},
                    {'Signal_Blocks', 'DSP System Toolbox'},
                    {'Distrib_Computing_Toolbox', 'Parallel Computing Toolbox'},
                    {'Communication_Toolbox','Communications System Toolbox'},
                    {'Bioinformatics_Toolbox','Bioinformatics Toolbox'},
                    {'Stateflow', 'Statistics and Machine Learning Toolbox'}];
                
[rtx ~] = size(required_toolboxes);

v = ver;

uninsTools = [];
count = 1;

for i = 1:rtx
    tool = required_toolboxes(i,:);
    %hasToolBox = license('test', tool{1});
    hasToolBox = any(strcmp(tool{2}, {v.Name}));
    if ~hasToolBox
        uninsTools{count} = tool{2};
        count = count + 1;
    else
        outText = strcat(tool{2}, ' already installed.');
        disp(outText);
    end
end


if count > 1
    outText = 'Sorry, but you do not seem to have the the following required toolboxes installed:';
    disp(outText);
    for i = 1:count - 1
        outText = strcat('    ', uninsTools{i});
        disp(outText);
    end
    outText = 'To install the required toolboxes on windows, login to your matlab account to purchase or download these toolboxes.';
    disp(outText);
    cd ..;
    return;
end

%--------------------
%Downloads matconvnet
%--------------------

disp('Downloading matconvnet... Please be patient');
urlwrite('https://github.com/vlfeat/matconvnet/archive/master.zip', 'matconvnet.zip');
disp('Downloaded! Unziping');
unzip('matconvnet.zip');
movefile('matconvnet-master', 'matconvnet');
delete('matconvnet.zip');

%--------------------
%File Copying
%--------------------
copyfile('+dagnn/BatchNorm2.m', 'matconvnet/matlab/+dagnn');
copyfile('+dagnn/Neg.m', 'matconvnet/matlab/+dagnn');
copyfile('+dagnn/Times.m', 'matconvnet/matlab/+dagnn');

%--------------------
%Installs matconvnet
%--------------------

%Sets up compilers
mex -setup C++;
mex -setup C;
cd matconvnet;

%Compiles/installs
addpath matlab;
vl_compilenn;
cd ..;

%Deletes setup.m in directory
delete('setup.m');

