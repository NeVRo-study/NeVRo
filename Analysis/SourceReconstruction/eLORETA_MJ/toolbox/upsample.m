function dataout=upsample(data,factor);

[ndat,ndum]=size(data);
ndathalf=ceil((ndat+1)/2);

df=fft(data);

if 2*(round(ndat/2))==ndat
   df=[df(1:ndathalf-1,:);df(ndathalf,:)/2;zeros((factor-1)*ndat-1,ndum);df(ndathalf,:)/2;df(ndathalf+1:end,:)];
else
    df=[df(1:ndathalf,:);zeros((factor-1)*ndat,ndum);df(ndathalf+1:end,:)];
end

dataout=ifft(df);

return;
