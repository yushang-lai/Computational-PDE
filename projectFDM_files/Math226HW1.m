%% Step 1: Uniform Grids and Grid Functions 
format long;
h     =   0.05;
hx    =   1;
hy    =   2;
[x,y] =   meshgrid(0:h:hx,hy:-2*h:0);
fun     =   myfunc(x,y);
figure(1);
surf(fun)


%% Step 2: Matrix-Vector Product

%**********************(I)********************

h      =   0.005;
hx     =   1;
hy     =   1;
[x,y]  =   meshgrid(0:h:hx,hy:-h:0);
u      =   myfunc(x,y);
u1     =   u(2:end-1,2:end-1);
us     =   u1(:); 
n      =   hx/h-1;
e      =   ones(n,1);
T      =   spdiags([-e 2*e -e], -1:1 ,n,n);
A      =   kron(speye(n),T)+kron(T,speye(n));

%*********************(II)********************

% um is the iteration matrix
um        =  zeros(n+2,n+2);
um(1,:)   =  u(1,:);
um(end,:) =  u(end,:);
um(:,1)   =  u(:,1);
um(:,end) =  u(:,end);

for k = 1:1
    j=2:n+1;
    i=2:n+1;
    um(i,j)=4*u(i,j)-u(i-1,j)-u(i+1,j)-u(i,j-1)-u(i,j+1);
end

uk = um(2:end-1,2:end-1);
um;
u  = myfunc(x,y);

%********************(III)*********************

B   =  (T*u1)*speye(n)+(speye(n)*u1)*T';
B   =  B(:);

f   =  8*pi^2*sin(2*pi*x).*cos(2*pi*y);
f   =  f(2:end-1,2:end-1);
f   =  f(:);

f(1) = f(1)+u(1,2)+u(2,1);
f(n) = f(n)+u(n+1,1)+u(n+2,2);
f(n*(n-1)+1) = f(n*(n-1)+1)+u(1,n+1)+u(2,n+2);
f(n*n) = f(n*n) + u(n+1,n+2)+u(n+2,n+1);

for i = 2:n-1
    f(i) = f(i)+ u(i+1,1);
end
for i = n*(n-1)+2:n^2-1
    f(i) = f(i)+ u(i+1-n*(n-1),n+2);
end
for i=1:n-2
    f(i*n+1) = f(i*n+1) + u(1,i+2);
end
for i=1:n-2
    f((i+1)*n) = f((i+1)*n) + u(n+2,i+2);
end

M   =  zeros((n-2)^2,3);

MM  =  reshape(A*us/h/h-f,n,n);
MM  =  MM(2:end-1,2:end-1);
MM  =  MM(:);

M(:,1)  = MM;

MM  =  reshape(uk(:)/h/h-f,n,n);
MM  =  MM(2:end-1,2:end-1);
MM  =  MM(:);

M(:,2)  = MM;

MM  =  reshape(B/h/h-f,n,n);
MM  =  MM(2:end-1,2:end-1);
MM  =  MM(:);

M(:,3)  = MM;

M1  = M;

% down h by half

h      =   h/2;
hx     =   1;
hy     =   1;
[x,y]  =   meshgrid(0:h:hx,hy:-h:0);
u      =   myfunc(x,y);
u1     =   u(2:end-1,2:end-1);
us     =   u1(:); 
n      =   hx/h-1;
e      =   ones(n,1);
T      =   spdiags([-e 2*e -e], -1:1 ,n,n);
A      =   kron(speye(n),T)+kron(T,speye(n));


%*********************(II)********************

% um is the iteration matrix
um        =  zeros(n+2,n+2);
um(1,:)   =  u(1,:);
um(end,:) =  u(end,:);
um(:,1)   =  u(:,1);
um(:,end) =  u(:,end);

for k = 1:1
    j=2:n+1;
    i=2:n+1;
    um(i,j)=4*u(i,j)-u(i-1,j)-u(i+1,j)-u(i,j-1)-u(i,j+1);
end

uk = um(2:end-1,2:end-1);
um;
u  = myfunc(x,y);

%********************(III)*********************

B   =  (T*u1)*speye(n)+(speye(n)*u1)*T';
B   =  B(:);

f   =  8*pi^2*sin(2*pi*x).*cos(2*pi*y);
f   =  f(2:end-1,2:end-1);
f   =  f(:);

f(1) = f(1)+u(1,2)+u(2,1);
f(n) = f(n)+u(n+1,1)+u(n+2,2);
f(n*(n-1)+1) = f(n*(n-1)+1)+u(1,n+1)+u(2,n+2);
f(n*n) = f(n*n) + u(n+1,n+2)+u(n+2,n+1);

for i = 2:n-1
    f(i) = f(i)+ u(i+1,1);
end
for i = n*(n-1)+2:n^2-1
    f(i) = f(i)+ u(i+1-n*(n-1),n+2);
end
for i=1:n-2
    f(i*n+1) = f(i*n+1) + u(1,i+2);
end
for i=1:n-2
    f((i+1)*n) = f((i+1)*n) + u(n+2,i+2);
end

M   =  zeros((n-2)^2,3);
MM  =  reshape(A*us/h/h-f,n,n);
MM  =  MM(2:end-1,2:end-1);
MM  =  MM(:);

M(:,1)  = MM;

MM  =  reshape(uk(:)/h/h-f,n,n);
MM  =  MM(2:end-1,2:end-1);
MM  =  MM(:);

M(:,2)  = MM;

MM  =  reshape(B/h/h-f,n,n);
MM  =  MM(2:end-1,2:end-1);
MM  =  MM(:);

M(:,3)  = MM;

M2  = M;

disp('The maximum norm of three method for h = 0.005');
disp(max(abs(M1)));
disp('The maximum norm of three method for h = 0.0025');
disp(max(abs(M2)));
disp('order of three methods');
disp(sqrt(max(abs(M1))./max(abs(M2))));

%% Step 3 Boundary Conditions Dirchlet

h     =   0.1;
hx    =   1;
hy    =   1;
[x,y] =   meshgrid(0:h:hx,hy:-h:0);
u     =   myfunc(x,y);
u1    =   u(2:end-1,2:end-1);
us    =   u1(:); 

n  = hx/h-1;
e  = ones(n,1);
T  = spdiags([-e 2*e -e], -1:1 ,n,n);
A  = kron(speye(n),T)+kron(T,speye(n));

f   =  h^2*8*pi^2*sin(2*pi*x).*cos(2*pi*y);
f   =  f(2:end-1,2:end-1);
f   =  f(:);

f(1) = f(1)+u(1,2)+u(2,1);
f(n) = f(n)+u(n+1,1)+u(n+2,2);
f(n*(n-1)+1) = f(n*(n-1)+1)+u(1,n+1)+u(2,n+2);
f(n*n) = f(n*n) + u(n+1,n+2)+u(n+2,n+1);

for i = 2:n-1
    f(i) = f(i)+ u(i+1,1);
end
for i = n*(n-1)+2:n^2-1
    f(i) = f(i)+ u(i+1-n*(n-1),n+2);
end
for i=1:n-2
    f(i*n+1) = f(i*n+1) + u(1,i+2);
end
for i=1:n-2
    f((i+1)*n) = f((i+1)*n) + u(n+2,i+2);
end

disp('By dirchlet method');
disp(reshape(A\f,n,n));
disp('Oringal matrix');
disp(u(2:end-1,2:end-1));
disp('Maximum diference');
disp(max(max(reshape(A\f,n,n)-u(2:end-1,2:end-1))));


%% Step 3 Boundary Conditions Neumann
h     =   0.025;
hx    =   1;
hy    =   1;
[x,y] =   meshgrid(0:h:hx,hy:-h:0);
u     =   myfunc(x,y); 
n     =   hx/h+1;

ff   =  h^2*8*pi^2*sin(2*pi*x).*cos(2*pi*y);
ff   =  ff(:);

fkx  =  @(x,y)  2*pi*cos(2*pi*x).*cos(2*pi*y);
fky  =  @(x,y) -2*pi*sin(2*pi*x).*sin(2*pi*y);

e  = ones(n,1);
T  = spdiags([-e 2*e -e], -1:1 ,n,n);
A  = kron(speye(n),T)+kron(T,speye(n));

A(1,1)   =  1;
A(1,2)   =  -0.5;
A(1,n+1) =  -0.5;
sz       = [n n];
[row,col] = ind2sub(sz,1);
ff(1)     = 0.25*ff(1)-h*(fkx(x(row,col),y(row,col))-fky(x(row,col),y(row,col)))/2;

A(n,n)   =  1;
A(n,n-1) =  -0.5;
A(n,2*n) =  -0.5;
[row,col] = ind2sub(sz,n);
ff(n)     = 0.25*ff(n)-h*(fkx(x(row,col),y(row,col))+fky(x(row,col),y(row,col)))/2;


A(n*(n-1)+1,n*(n-1)+1) = 1;
A(n*(n-1)+1,n*(n-1)+2) = -0.5;
A(n*(n-1)+1,n*(n-2)+1) = -0.5;
[row,col] = ind2sub(sz,n*(n-1)+1);
ff(n*(n-1)+1)     = 0.25*ff(n*(n-1)+1)+h*(fkx(x(row,col),y(row,col))+fky(x(row,col),y(row,col)))/2;

A(n^2,n^2)     = 1;
A(n^2,n^2-1)   = -0.5;
A(n^2,n*(n-1)) = -0.5;
[row,col] = ind2sub(sz,n^2);
ff(n^2)         = 0.25*ff(n^2)+h*(fkx(x(row,col),y(row,col))-fky(x(row,col),y(row,col)))/2;

for i = 2:n-1
    A(i,i) = 2;
    A(i,i-1) = -0.5;
    A(i,i+1) = -0.5;
    A(i,i+n) = -1;
    [row,col] = ind2sub(sz,i);
    ff(i)     = 0.5*ff(i) - h*fkx(x(row,col),y(row,col));
end

for i = n*(n-1)+2:n^2-1
    A(i,i) = 2;
    A(i,i-1) = -0.5;
    A(i,i+1) = -0.5;
    A(i,i-n) = -1;
    [row,col] = ind2sub(sz,i);
    ff(i)     = 0.5*ff(i) + h*fkx(x(row,col),y(row,col));
end

for i=1:n-2
    A(i*n+1,i*n+1)   = 2;
    A(i*n+1,i*n+1-n) = -0.5;
    A(i*n+1,i*n+1+n) = -0.5;
    A(i*n+1,i*n+1+1) = -1;
    [row,col] = ind2sub(sz,i*n+1);
    ff(i*n+1)     = 0.5*ff(i*n+1) - h*fky(x(row,col),y(row,col));
    
end

for i=1:n-2
    A((i+1)*n,(i+1)*n) = 2;
    A((i+1)*n,(i+1)*n-n) = -0.5;
    A((i+1)*n,(i+1)*n+n) = -0.5;
    A((i+1)*n,(i+1)*n-1) = -1;
    [row,col] = ind2sub(sz,(i+1)*n);
    ff((i+1)*n)     = 0.5*ff((i+1)*n) + h*fky(x(row,col),y(row,col));
end

A;
D = u;

BB = reshape(pinv(full(A))*ff,n,n);
BB(1:10,1:10)
u(1:10,1:10)


disp('By Nemann method first 10*10');
disp(BB(1:10,1:10));
disp('Oringal matrix first 10*10');
disp(u(1:10,1:10));
disp('Maximum diference');
disp(max(max(BB(1:10,1:10)-u(1:10,1:10))));


%% Step 4 Solve the linear Algebraic Systems

tol   =   2*10^(-2);
err   =   1;
h     =   0.1;
hx    =   1;
hy    =   1;
[x,y] =   meshgrid(0:h:hx,hy:-h:0);
u     =   myfunc(x,y);
u1    =   u(2:end-1,2:end-1);
us    =   u1(:); 

n  = hx/h-1;
e  = ones(n,1);
T  = spdiags([-e 2*e -e], -1:1 ,n,n);
A  = kron(speye(n),T)+kron(T,speye(n));

f   =  h^2*8*pi^2*sin(2*pi*x).*cos(2*pi*y);
f   =  f(2:end-1,2:end-1);
f   =  f(:);

f(1) = f(1)+u(1,2)+u(2,1);
f(n) = f(n)+u(n+1,1)+u(n+2,2);
f(n*(n-1)+1) = f(n*(n-1)+1)+u(1,n+1)+u(2,n+2);
f(n*n) = f(n*n) + u(n+1,n+2)+u(n+2,n+1);

for i = 2:n-1
    f(i) = f(i)+ u(i+1,1);
end
for i = n*(n-1)+2:n^2-1
    f(i) = f(i)+ u(i+1-n*(n-1),n+2);
end
for i=1:n-2
    f(i*n+1) = f(i*n+1) + u(1,i+2);
end
for i=1:n-2
    f((i+1)*n) = f((i+1)*n) + u(n+2,i+2);
end
disp('Dierect Solving')
disp(reshape(A\f,n,n));
disp('Original matrix')
disp(u(2:end-1,2:end-1));
disp('Maximum diference');
disp(max(max(reshape(A\f,n,n)-u(2:end-1,2:end-1))));

f = reshape(f,n,n);
um = zeros(n,n);
u2 = u(2:end-1,2:end-1);
um(1,:)= u2(1,:);
um(end,:)=u2(end,:);
um(:,1)=u2(:,1);
um(:,end) = u2(:,end);

 while err > tol
     for j = 2:n-1
         for i = 2:n-1
            um(i,j)  = (f(i,j) + um(i-1,j) + um(i+1,j) + um(i,j-1) +um(i,j+1))/4;
         end
     end
     err = norm(f(:)-A*um(:))/norm(f(:));
 end

 disp('when tol is 2*e3')
 disp('GS Solving')
 disp(um);
 disp('Original Matrix')
 disp(u(2:end-1,2:end-1));
 disp('Maximum diference');
 disp(max(max(um-u(2:end-1,2:end-1))));
 
 
 %% Step 5 Convergence 
 
 lst = [];
for k = [1 8 16 32 64]

    h      =   0.05/k;
    hx     =   1;
    hy     =   1;
    [x,y]  =   meshgrid(0:h:hx,hy:-h:0);
    u      =   myfunc(x,y);
    u1     =   u(2:end-1,2:end-1);
    us     =   u1(:); 
    n      =   hx/h-1;
    e      =   ones(n,1);
    T      =   spdiags([-e 2*e -e], -1:1 ,n,n);
    A      =   kron(speye(n),T)+kron(T,speye(n));
    f   =  8*pi^2*sin(2*pi*x).*cos(2*pi*y);
    f   =  f(2:end-1,2:end-1);
    f   =  f(:);



    f(1) = f(1)+u(1,2)+u(2,1);
    f(n) = f(n)+u(n+1,1)+u(n+2,2);
    f(n*(n-1)+1) = f(n*(n-1)+1)+u(1,n+1)+u(2,n+2);
    f(n*n) = f(n*n) + u(n+1,n+2)+u(n+2,n+1);

    for i = 2:n-1
        f(i) = f(i)+ u(i+1,1);
    end
    for i = n*(n-1)+2:n^2-1
        f(i) = f(i)+ u(i+1-n*(n-1),n+2);
    end
    for i=1:n-2
        f(i*n+1) = f(i*n+1) + u(1,i+2);
    end
    for i=1:n-2
        f((i+1)*n) = f((i+1)*n) + u(n+2,i+2);
    end

    
    MM  =  reshape(A*us/h/h-f,n,n);
    MM  =  MM(2:end-1,2:end-1);
    MM  =  max(MM(:));
    lst = [lst MM];
end

disp('when I down the divide difference by 1/8 1/16 1/32  1/64');
disp('it equals to around 8 16 32 64');
disp(sqrt((lst(1)./lst(2:end))));
disp('Thus the oreder is 2;
