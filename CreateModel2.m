function model=CreateModel2(DamageRatio,DamageLocation)

Coor=[0 0 2 2 0 0 2 2 0 0 2 2
      0 2 0 2 0 2 0 2 0 2 0 2
      0 0 0 0 2 2 2 2 4 4 4 4]';

Conn=[1 5; 2 6; 3 7; 4 8; 5 9; 6 10; 7 11; 8 12;
      5 6; 7 8; 9 10; 11 12; 5 7; 6 8; 9 11; 10 12];

A=ones(size(Conn,1))*0.0075;

BC=1:6*4;

Ro=7800;

E=ones(size(Conn,1))*210e9;
E(DamageLocation)=E(DamageLocation)*(1-DamageRatio);

v=0.3;

model.Conn=Conn;
model.Coor=Coor;
model.BC=BC;
model.A=A;
model.Ro=Ro;
model.E=E;
model.v=v;
model.I=1;
model.J=model.I;
model.G=model.E;
