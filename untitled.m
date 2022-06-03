sum2=0;
M=2;N=2;
F= 2*ones(M,N)
for k=1:M% first summation
    % row wise
    sum=0;
    
    for l= 1:N % second summation
        % column wise 
%        F(k,l)=  F(k,l)* F(k,l);
       sum= sum+ F(k,l)* exp(l*2*pi);
        
    end
    sum = sum *exp(k*2*pi);
    sum2= sum2+ sum; 
end

mod= imag(sum2)^2 + real(sum2)^2;

rand_cmplx= rand(3,3) +rand(3,3).*i