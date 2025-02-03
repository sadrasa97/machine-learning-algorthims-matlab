function model=CreateModel3(DamageRatio,DamageLocation)

Coor=[0 0 0;1.52 0 0;3.04 0 0;4.56 0 0;6.08 0 0;7.6 0 0;9.12 0 0;0 1.52 0;1.52 1.52 0;3.04 1.52 0;4.56 1.52 0;6.08 1.52 0;7.6 1.52 0;9.12 1.52 0];

Conn=[1 8;8 9;2 8;1 9;1 2;2 9;9 10;3 9;2 10;2 3;3 10;10 11;4 10;3 11;3 4;4 11;11 12;5 11;4 12;4 5;5 12;12 13;6 12;5 13;5 6;6 13;13 14;7 13;6 14;6 7;7 14];


A=ones(size(Conn,1))*1.8e-3;

BC=[1 2 20 3 6 9 12 15 18 21 24 27 30 33 36 39 42];
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
