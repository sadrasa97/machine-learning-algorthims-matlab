function PlotFrame(model,U,n,C)
%% Plot
U=U(:,n)/max(abs(U(:,n)))*0.2;
Coor=model.Coor;
Conn=model.Conn;
ne=size(Conn,1);

figure(20)
set(gcf,'Position',[100 100 1000 600])
hold on
ScaleFactor=1;
for i=1:ne
    node1=Conn(i,1);
    node2=Conn(i,2);
    
    x1=Coor(node1,1)+U(6*node1-5)*ScaleFactor;
    x2=Coor(node2,1)+U(6*node2-5)*ScaleFactor;
    y1=Coor(node1,2)+U(6*node1-4)*ScaleFactor;
    y2=Coor(node2,2)+U(6*node2-4)*ScaleFactor;   
    z1=Coor(node1,3)+U(6*node1-3)*ScaleFactor;
    z2=Coor(node2,3)+U(6*node2-3)*ScaleFactor;
    
    plot3([x1 x2],[y1 y2],[z1 z2],'--','LineWidth',2.5,'Color',C);

    text(0.6*x1+0.4*x2,0.6*y1+0.4*y2,0.6*z1+0.4*z2,num2str(i),'FontSize',12)
    
    plot3(x1,y1,z1,'ob','MarkerFaceColor',C)
    plot3(x2,y2,z2,'ob','MarkerFaceColor',C)
    view([0 90])
end

ScaleFactor=0;
for i=1:ne
    node1=Conn(i,1);
    node2=Conn(i,2);
    
    x1=Coor(node1,1)+U(6*node1-5)*ScaleFactor;
    x2=Coor(node2,1)+U(6*node2-5)*ScaleFactor;
    y1=Coor(node1,2)+U(6*node1-4)*ScaleFactor;
    y2=Coor(node2,2)+U(6*node2-4)*ScaleFactor;   
    z1=Coor(node1,3)+U(6*node1-3)*ScaleFactor;
    z2=Coor(node2,3)+U(6*node2-3)*ScaleFactor;
    
    plot3([x1 x2],[y1 y2],[z1 z2],'k','LineWidth',2.5);

    text(0.6*x1+0.4*x2,0.6*y1+0.4*y2,0.6*z1+0.4*z2,num2str(i),'FontSize',12)
    
    plot3(x1,y1,z1,'ok','MarkerFaceColor','k')
    plot3(x2,y2,z2,'ok','MarkerFaceColor','k')
    text(x1,y1-0.1,z1-0.1,num2str(node1),'FontSize',10)
    text(x2,y2-0.1,z2-0.1,num2str(node2),'FontSize',10)
    axis equal
    if all(Coor(:,3)==0)
        view([0 90])
    else
        view([30 20])
    end
end