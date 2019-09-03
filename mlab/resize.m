close all;clear all;clc;
for i=1:50
    rfilename=['D:\project\viola_jones\ogim\training (',num2str(i),').bmp'];
    wfilename=['D:\project\viola_jones\training_set\training',num2str(i),'.bmp'];
    I0=imread(rfilename);
    I1=imresize(I0,[32 32]);
    imwrite(I1,wfilename);
end

for i=1:24
    rfilename=['D:\project\viola_jones\ogim\testing (',num2str(i),').bmp'];
    wfilename=['D:\project\viola_jones\testing_set\testing',num2str(i),'.bmp'];
    I0=imread(rfilename);
    I1=imresize(I0,[32 32]);
    imwrite(I1,wfilename);
end