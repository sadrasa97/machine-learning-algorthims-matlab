function model=CreateModel1(DamageRatio,DamageLocation)

Coor=[0 2 4 6 8 10 12 2 4 6 8 10;
      0 0 0 0 0 0 0 1.16 2.2 2.6 2.2 1.16;
      0 0 0 0 0 0 0 0 0 0 0 0]';
Conn=[1 8; 8 9; 9 10; 10 11; 11 12; 7 11; 1 2; 2 3; 3 4
      4 5; 5 6; 6 7; 2 8; 3 9; 4 10; 5 11; 6 12; 2 9; 3 10
      5 10; 5 12; 3 8; 4 9; 4 11; 6 11];

A=ones(size(Conn,1))*1.8e-3;

BC=[1 2 20 3:3:36];
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
