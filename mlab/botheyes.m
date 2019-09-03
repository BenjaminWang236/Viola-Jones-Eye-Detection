function [location]=botheyes(img_32x32)
	I1=imresize(img_32x32,4);
	%eyedetect=vision.CascadeObjectDetector('EyePairBig');
	eyedetect=vision.CascadeObjectDetector('EyePairSmall');
	bbox=eyedetect(I1);
    ss=size(bbox);
        
    if ss(1,1)==0
        location=[0 0 0 0];
    elseif ss(1,1)>1         
        [val,idx]=min(bbox);
        nn=idx(1,1);
        bbox1=round(bbox(nn,:)./4);
        location=bbox1;
    else
        nn=1;
        bbox1=round(bbox(nn,:)./4);
        location=bbox1;
    end
    
    


	%location(1:2)=bbox1(1:2);
    %location(3)=bbox1(1)+bbox1(3);
    %location(4)=bbox1(2)+bbox1(4);
end