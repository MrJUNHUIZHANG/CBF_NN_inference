function y=RBF(x,centers,K)
spread=0.8326;
H=[];
for i=1:K
    h=exp(-spread^2*sum((x-centers(:,i)).^2));
    H=[H;h];
end
y=[H;1];
end