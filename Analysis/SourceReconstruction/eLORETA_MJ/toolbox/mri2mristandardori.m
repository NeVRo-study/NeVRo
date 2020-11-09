function [data_out,scales_out]=mri2mristandardori(data,ori,scales);
%
% transforms MRI-tensor data 
% into standard form, i.e. first index is 
% from right to left ear, second index from 
% front to back, and 3rd index from bottom to top. 
%
% INPUT:
% ori is a vector containing  the 3 numbers 1,2, 3
% (eventually also negative) to indicate the meaning 
% of the indices in data. ori(1) says what index 
% in data refers to the standard x-axis (i.e. right ear to left 
% ear)  etc..
% ori(2) denotes the index in data refering to standard 
% y axis (front to back), and ori(3) denotes the index refering 
% to bottom to top axis.  
% If any of the numbers is negative the resespective order 
% is in the wrong direction. 
% 
% Excamples:
%  ori=[1 2 3] means that the data are already in correct order;
%
%  ori=[2 3 1] means that the 2nd index of data is from right 
%    ear to left ear, the 3rd  index is from front to back, and the 
%    first index is from bottom to to top.
%
%  ori=[3 1 -2] means that the 3rd index of data is from right ear 
%    to left ear, the 1st index is from front to back, and the 
%    2nd index is from top to bottom. 
%
% scales: is a 1x3 vector denoting the physical voxel-distance 
% in centimeters for the i.th index in data. 
%
% output:
% data_out:  % MRI-data in correct order. 
% scales_out: scales in correct order. 


nn=size(data);

%switch ori
%  case 'fbtdlr'
%    data_out=zeros(nz,nx,ny);
%    for i=1:nz
%      data_out(nz-i+1,:,:)=fliplr(squeeze(data(:,:,i)));
%    end
%end

nx=nn(abs(ori(1)));
ny=nn(abs(ori(2)));
nz=nn(abs(ori(3)));

data_out=zeros(nx,ny,nz);
  
  for ix=1:nx;
    switch abs(ori(1))
      case 1  
        data_loc=squeeze(data(ix,:,:));
      case 2
        data_loc=squeeze(data(:,ix,:));   
      case 3
        data_loc=squeeze(data(:,:,ix));   
      case -1  
        data_loc=squeeze(data(nx-ix+1,:,:));
      case -2
        data_loc=squeeze(data(:,nx-ix+1,:));   
      case -3
        data_loc=squeeze(data(:,:,nx-ix+1)); 
    end
    for iy=1:ny 
      if ori(2)>0
        if abs(ori(2))<abs(ori(3))
          data_locloc=data_loc(iy,:)';
        else 
          data_locloc=data_loc(:,iy);
        end
      else 
        if abs(ori(2))<abs(ori(3))
          data_locloc=data_loc(ny-iy+1,:)';
        else 
          data_locloc=data_loc(:,ny-iy+1);
        end
      end
       if ori(3)<0
            data_locloc=flipud(data_locloc);
      end

      data_out(ix,iy,:)=data_locloc;
    end
  end

scales_out=scales(abs(ori));

return;
