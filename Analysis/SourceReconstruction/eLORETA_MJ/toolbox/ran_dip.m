function dip=ran_dip(center,radius,ndip,type,bias);

[n1,m1]=size(center);
if m1==3 & n1==1
    center=center';
end

    if nargin==5
      [n1,m1]=size(bias);
      if m1==3 & n1==1
        bias=bias';
      end
      center_b=center+bias;
    else
      center_b=center;
    end


if type=='cart'
    
  diploc=[];
  for i=1:ndip
    nn=2*radius;
     while nn>radius; 
         xx=2*radius*(rand(1,3)-.5);
         nn=norm(xx); 
     end; 
     diploc=[diploc;xx+center_b'];
  end
  
  dip=[diploc,randn(ndip,3)];
  
elseif type=='sphe'
    
    dip=[];
    
    for i=1:ndip
      nn=2*radius;
      while nn>radius; 
         xx=2*radius*(rand(1,3)-.5);
         nn=norm(xx); 
      end; 
        dip=[dip;xx'+center_b;pi*rand(1,1);2*pi*rand(1,1)];
    end
    
end

return;