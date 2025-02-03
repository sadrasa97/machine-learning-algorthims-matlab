function [NaturalFrequancy,ModeShape,U]=TrussFEM(model,nF)

Coor=model.Coor;
Conn=model.Conn;
A=model.A;
E=model.E;
Ro=model.Ro;
BC=model.BC;


nn=size(Coor,1);
ne=size(Conn,1);
notBC=setdiff(1:3*nn,BC);

%% Local Stiffness and Mass
Le=zeros(ne,1);
l=zeros(ne,1);
m=zeros(ne,1);
n=zeros(ne,1);
K=zeros(6,6,ne);
M=zeros(6,6,ne);

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
    l(i)=(x2-x1)/Le(i);                                  %(4.6)
    m(i)=(y2-y1)/Le(i);                                  %(4.6)
    n(i)=(z2-z1)/Le(i);                                  %(4.6)
    
    L=[l(i) m(i) n(i) 0 0 0 ; 0 0 0 l(i) m(i) n(i)];     %(4.5)
    
    k=E(i)*A(i)/Le(i)*[1 -1; -1 1];                         %(4.8)
    K(:,:,i)=L'*k*L;                                     %(4.12)
    mm=Ro*A(i)*Le(i)/6*[2 1;1 2];
    M(:,:,i)=L'*mm*L;  
end
    
%% Assembly Global Stiffness and Mass
ndof=3*nn;
KG=zeros(ndof,ndof);
MG=zeros(ndof,ndof);

for i=1:ne
    
    node1=Conn(i,1);
    node2=Conn(i,2);
    
    dof=[3*node1-2 3*node1-1 3*node1 3*node2-2 3*node2-1 3*node2];
    KG(dof,dof)=KG(dof,dof)+K(:,:,i);
    MG(dof,dof)=MG(dof,dof)+M(:,:,i);
end
    
%% Solution Equations 
[ModeShape,NaturalFrequancy] = eig(KG(notBC,notBC),MG(notBC,notBC));
[NaturalFrequancy,so]=sort(diag(NaturalFrequancy));
ModeShape=ModeShape(so,:);    

U=zeros(3*nn,8);
for i=1:8
    U(notBC,i)=ModeShape(i,:)';
end
if ~isempty(nF)
    NaturalFrequancy=NaturalFrequancy(1:nF);
end
