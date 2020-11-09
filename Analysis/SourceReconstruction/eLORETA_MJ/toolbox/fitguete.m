function [fitg, am]=fitguete(a,b);
% a measuered, 
% b model 

am=(a'*b)*b/(b'*b);

fitg=norm(a-am)/norm(a);

return;
