function [NaturalFrequancy,ModeShape,U]=FrameFEM(model,nF)

Coor=model.Coor;
Conn=model.Conn;
A=model.A;
E=model.E;
Ro=model.Ro;
BC=model.BC;
I=model.I;
J=model.J;
G=model.G;

nn=size(Coor,1);
ne=size(Conn,1);
notBC=setdiff(1:6*nn,BC);

%% Local Stiffness and Mass
Le=zeros(ne,1);
K=zeros(12,12,ne);
M=zeros(12,12,ne);

for i=1:ne

    node1=Conn(i,1);
    node2=Conn(i,2);

    x1=Coor(node1,1);
    x2=Coor(node2,1);
    y1=Coor(node1,2);
    y2=Coor(node2,2);
    z1=Coor(node1,3);
    z2=Coor(node2,3);


    Le(i)=((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)^0.5;       %(4.7)
    C1=A(i)*E(i)/Le(i);
    C2=12*E(i)*I/(Le(i)^3);
    C3=6*E(i)*I/(Le(i)^2);
    C4=4*E(i)*I/Le(i);
    C5=C4/2;
    C6=G(i)*J/Le(i);


    l=(x2-x1)/Le(i);
    m=(y2-y1)/Le(i);
    n=(z2-z1)/Le(i);

    if x1 == x2 && y1 == y2
        if z2 > z1
            Landa = [0 0 1;
                0 1 0;
                -1 0 0];  %This assumes EOV is the global y axis
        else
            Landa = [0 0 -1;
                0 1 0;
                1 0 0];  %This assumes EOV is the global y axis
        end
    else

        D=sqrt(l^2+m^2);
            if D==0
                D=n;
            end

        Landa=[l m n;-m/D l/D 0; -l*n/D -m*n/D D];
    end
    T=zeros(12);
    T(1:3,1:3)=Landa;
    T(4:6,4:6)=Landa;
    T(7:9,7:9)=Landa;
    T(10:12,10:12)=Landa;


    k=[C1 0 0 0 0 0 -C1 0 0 0 0 0
        0 C2 0 0 0 C3 0 -C2 0 0 0 C3
        0 0 C2 0 -C3 0 0 0 -C2 0 -C3 0
        0 0 0 C6 0 0 0 0 0 -C6 0 0
        0 0 -C3 0 C4 0 0 0 C3 0 C5 0
        0 C3 0 0 0 C4 0 -C3 0 0 0 C5
        -C1 0 0 0 0 0 C1 0 0 0 0 0
        0 -C2 0 0 0 -C3 0 C2 0 0 0 -C3
        0 0 -C2 0 C3 0 0 0 C2 0 C3 0
        0 0 0 -C6 0 0 0 0 0 C6 0 0
        0 0 -C3 0 C5 0 0 0 C3 0 C4 0
        0 C3 0 0 0 C5 0 -C3 0 0 0 C4];

    a=Le(i)/2;
    r=(I/A(i))^0.5;
    mm=Ro*A(i)*a/105*[70 0 0 0 0 0 35 0 0 0 0 0
        0 78 0 0 0 22*a 0 27 0 0 0 -13*a
        0 0 78 0 -22*a 0 0 0 27 0 13*a 0
        0 0 0 7*r^2 0 0 0 0 0 -35*r^2 0 0
        0 0 -22*a 0 8*a^2 0 0 0 -13*a 0 -6*a^2 0
        0 22*a 0 0 0 8*a^2 0 13*a 0 0 0 -6*a^2
        35 0 0 0 0 0 70 0 0 0 0 0
        0 27 0 0 0 13*a 0 78 0 0 0 -22*a
        0 0 27 0 -13*a 0 0 0 78 0 22*a 0
        0 0 0 -35*r^2 0 0 0 0 0 70*r^2 0 0
        0 0 13*a 0 -6*a^2 0 0 0 22*a 0 8*a^2 0
        0 -13*a 0 0 0 -6*a^2 0 -22*a 0 0 0 8*a^2];

    K(:,:,i)=T'*k*T;                                     %(4.12)
    M(:,:,i)=T'*mm*T;
end

%% Assembly Global Stiffness and Mass
ndof=6*nn;
KG=zeros(ndof,ndof);
MG=zeros(ndof,ndof);

for i=1:ne

    node1=Conn(i,1);
    node2=Conn(i,2);

    dof=[6*node1-5 6*node1-4 6*node1-3 6*node1-2 6*node1-1 6*node1...
        6*node2-5 6*node2-4 6*node2-3 6*node2-2 6*node2-1 6*node2];
    KG(dof,dof)=KG(dof,dof)+K(:,:,i);
    MG(dof,dof)=MG(dof,dof)+M(:,:,i);
end

%% Solution Equations
[ModeShape,NaturalFrequancy] = eig(KG(notBC,notBC),MG(notBC,notBC));
[NaturalFrequancy,so]=sort(diag(NaturalFrequancy));
ModeShape=ModeShape(so,:);

U=zeros(6*nn,8);
for i=1:8
    U(notBC,i)=ModeShape(i,:)';
end
if ~isempty(nF)
    NaturalFrequancy=NaturalFrequancy(1:nF);
end
