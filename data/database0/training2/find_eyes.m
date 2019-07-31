close all; clear all;
samples=50;
%eye_point=zeros(samples,4);
for i=1:samples
    rfilename=['D:\Ben Wang\OneDrive\NeuronBasic\Viola-Jones-Eye-Detection\data\database0\training2\2training',num2str(i),'.bmp'];
    I0=imread(rfilename);
    eye_point(i,:)=botheyes(I0);
end
pp=uint8(eye_point);
fid = fopen('D:\Ben Wang\OneDrive\NeuronBasic\Viola-Jones-Eye-Detection\data\database0\training2\eye_point.bin', 'w', 'ieee-le');
fwrite(fid, pp, 'uint8');
fclose(fid);

%fid = fopen('D:\project\viola_jones\training_set\eye_point.bin', 'r', 'ieee-le');
%tt=fread(fid, [2 2 19], 'uint8');
%fclose(fid);

%multibandwrite(pp,'D:\project\viola_jones\training_set\eye_point.bin','bil');
%tt=multibandread('D:\project\viola_jones\training_set\eye_point.bin',[2 2 samples],'uint8=>uint8',0,'bip','ieee-le');