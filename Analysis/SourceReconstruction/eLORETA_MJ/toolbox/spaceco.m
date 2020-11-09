function s=spaceco(A,B);

P1=A*inv(A'*A)*A';
P2=B*inv(B'*B)*B';
P=P1*P2*P1;

[u s v]=svd(P);
s=diag(s);

return