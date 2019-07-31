%%
clear all;close all;
%rfilename=['D:\project\ogim\Person_right&left_80lux_side_50lux_big.bmp'];
rfilename=['D:\project\ogim\Mannequin_right&left_10lux_side_10lux_big.bmp'];
I=imread('rfilename');
I1=imresize(I,[64 64]);
I2=imresize(I,[32 32]);
%I2=histeq(I2);
subplot(2,2,1);imshow(I2);
title("Original");
%%
%eyedetect=vision.CascadeObjectDetector('EyePairBig');
eyedetect=vision.CascadeObjectDetector('EyePairSmall');
bbox=eyedetect(I1);
bbox2=round(bbox./2);
%eyes=insertObjectAnnotation(I,'rectangle',bbox,'Both Eyes');
eyes=insertObjectAnnotation(I2,'rectangle',bbox2,'B');
botheyes=I1(bbox(2):bbox(2)+bbox(4),bbox(1):bbox(1)+bbox(3));
botheyes=imresize(botheyes,2);
subplot(2,2,2);imshow(eyes);
my_title=['(',num2str(bbox2(2)),',',num2str(bbox2(1)),') (',num2str(bbox2(2)+bbox2(4)),',',num2str(bbox2(1)+bbox2(3)),')'];
title(my_title);
%title("("+bbox2(1)+","+bbox2(2)+") ("+(bbox2(1)+bbox2(3))+","+(bbox2(2)+bbox2(4))+")");
%%
eyedetect=vision.CascadeObjectDetector('LeftEye');
lbox=eyedetect(botheyes);
%llbox=lbox(1,:);
ss=size(lbox);
if ss(1,1)>1
    [val,idx]=min(lbox);
    nn=idx(1,1);
else
    nn=1;
end
llbox=round(lbox(nn,:)./4);
llbox(1:2)=(llbox(1:2)+bbox2(1:2));
%left=insertObjectAnnotation(I,'rectangle',llbox,'Left Eye');
left=insertObjectAnnotation(I2,'rectangle',llbox,'L');
subplot(2,2,3),imshow(left);
my_title=['(',num2str(llbox(2)),',',num2str(llbox(1)),') (',num2str(llbox(2)+llbox(4)),',',num2str(llbox(1)+llbox(3)),')'];
title(my_title);
%title("("+llbox(1)+","+llbox(2)+") ("+(llbox(1)+llbox(3))+","+(llbox(2)+llbox(4))+")");
%%
eyedetect=vision.CascadeObjectDetector('RightEye');
rbox=eyedetect(botheyes);
%rrbox=rbox(1,:);
ss=size(rbox);
if ss(1,1)>1
    [val,idx]=max(rbox);
    nn=idx(1,1);
else
    nn=1;
end
rrbox=round(rbox(nn,:)./4);
rrbox(1:2)=rrbox(1:2)+bbox2(1:2);
%right=insertObjectAnnotation(I,'rectangle',rrbox,'Right Eye');
right=insertObjectAnnotation(I2,'rectangle',rrbox,'R');
subplot(2,2,4),imshow(right);
my_title=['(',num2str(rrbox(2)),',',num2str(rrbox(1)),') (',num2str(rrbox(2)+rrbox(4)),',',num2str(rrbox(1)+rrbox(3)),')'];
title(my_title);
%title("("+rrbox(1)+","+rrbox(2)+") ("+(rrbox(1)+rrbox(3))+","+(rrbox(2)+rrbox(4))+")");