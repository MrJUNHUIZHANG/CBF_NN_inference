tt=dt*(0:1:length(W1_online)-1);

% %====== plot prediction interval=============

figure(2)
fill([tt, fliplr(tt)], [ua_ub(1,:), fliplr(ua_lb(1,:))], 'b', 'FaceAlpha', 0.2); % 'c'
hold on 
plot(tt,u_a1(1,:),'b','lineWidth',1.2)
hold on
plot(tt,ua_est(1,:),'r','lineWidth',1.2)
xlabel('Time t(s)','Fontname','Times New Roman','FontSize', 13);
ylabel({'Dynamics $\dot{p}_{x,1}$'},'Fontname','Times New Roman','FontSize', 13,'Interpreter', 'latex');
title('Agent 1 in the x direction','Fontname','Times New Roman','FontSize', 13);
legend('Prediction interval','Real dynamics','RBFNN estimate','Fontname','Times New Roman','FontSize', 13);
axis([0 30 -1.5 2])

figure(3)
fill([tt, fliplr(tt)], [ua_ub(2,:), fliplr(ua_lb(2,:))], 'b', 'FaceAlpha', 0.2); % 'c'
hold on 
plot(tt,u_a1(2,:),'b','lineWidth',1.2)
hold on
plot(tt,ua_est(2,:),'r','lineWidth',1.2)
xlabel('Time t(s)','Fontname','Times New Roman','FontSize', 13);
ylabel({'Dynamics $\dot{p}_{y,1}$'},'Fontname','Times New Roman','FontSize', 13,'Interpreter', 'latex');
title('Agent 1 in the y direction','Fontname','Times New Roman','FontSize', 13);
legend('Prediction interval','Real dynamics','RBFNN estimate','Fontname','Times New Roman','FontSize', 13);
axis([0 30 -1.5 2])

figure(4)
fill([tt, fliplr(tt)], [ua_ub(3,:), fliplr(ua_lb(3,:))], 'b', 'FaceAlpha', 0.2); % 'c'
hold on 
plot(tt,u_a2(1,:),'b','lineWidth',1.2)
hold on
plot(tt,ua_est(3,:),'r','lineWidth',1.2)
xlabel('Time t(s)','Fontname','Times New Roman','FontSize', 13);
ylabel({'Dynamics $\dot{p}_{x,2}$'},'Fontname','Times New Roman','FontSize', 13,'Interpreter', 'latex');
title('Agent 2 in the x direction','Fontname','Times New Roman','FontSize', 13)
legend('Prediction interval','Real dynamics','RBFNN estimate','Fontname','Times New Roman','FontSize', 13);
axis([0 30 -1.5 2])

figure(5)
fill([tt, fliplr(tt)], [ua_ub(4,:), fliplr(ua_lb(4,:))], 'b', 'FaceAlpha', 0.2); % 'c'
hold on 
plot(tt,u_a2(2,:),'b','lineWidth',1.2)
hold on
plot(tt,ua_est(4,:),'r','lineWidth',1.2)
xlabel('Time t(s)','Fontname','Times New Roman','FontSize', 13);
ylabel({'Dynamics $\dot{p}_{y,2}$'},'Fontname','Times New Roman','FontSize', 13,'Interpreter', 'latex');
title('Agent 2 in the y direction','Fontname','Times New Roman','FontSize', 13)
legend('Prediction interval','Real dynamics','RBFNN estimate','Fontname','Times New Roman','FontSize', 13);
axis([0 30 -1.5 2])

%-------------------- Compare with online/offline learning ------------------------------
figure(6)
tt=dt*(0:1:length(ua_est)-1);
plot(tt,sqrt((u_a1(1,:)-ua_est(1,:)).^2+(u_a1(2,:)-ua_est(2,:)).^2),'b','lineWidth',1.2)
hold on
plot(tt,sqrt((u_a1(1,:)-ua_offline(1,:)).^2+(u_a1(2,:)-ua_offline(2,:)).^2),'k','lineWidth',1.2)
hold on
plot(tt,sqrt((u_a1(1,:)-ua_online(1,:)).^2+(u_a1(2,:)-ua_online(2,:)).^2),'m','lineWidth',1.2)
xlabel('Time t(s)','Fontname','Times New Roman','FontSize', 13);
ylabel('Estimation error $\|\hat{\dot{p}}_{1}-\dot{p}_{1}\|$','Fontname','Times New Roman','FontSize', 13,'Interpreter', 'latex');
legend('Estimation by offline-online inference','Estimation by offline learning', 'Estimation by online learning','Fontname','Times New Roman','FontSize', 13);
axis([0 30 0 1])

figure(7)
tt=dt*(0:1:length(ua_est)-1);
plot(tt,sqrt((u_a2(1,:)-ua_est(3,:)).^2+(u_a2(2,:)-ua_est(4,:)).^2),'b','lineWidth',1.2)
hold on
plot(tt,sqrt((u_a2(1,:)-ua_offline(3,:)).^2+(u_a2(2,:)-ua_offline(4,:)).^2),'k','lineWidth',1.2)
hold on
plot(tt,sqrt((u_a2(1,:)-ua_online(3,:)).^2+(u_a2(2,:)-ua_online(4,:)).^2),'m','lineWidth',1.2)

xlabel('Time t(s)','Fontname','Times New Roman','FontSize', 13);
ylabel('Estimation error $\|\hat{\dot{p}}_{2}-\dot{p}_{2}\|$','Fontname','Times New Roman','FontSize', 13,'Interpreter', 'latex');
legend('Estimation by offline-online inference','Estimation by offline learning', 'Estimation by online learning','Fontname','Times New Roman','FontSize', 13);
axis([0 30 0 0.5])

% Plot width 
figure
tt=dt*(0:1:length(ua_est)-1);
plot(tt,ua_ub(1,:)-ua_est(1,:),'lineWidth',1.2,'color','r');
hold on
plot(tt,ua_ub(4,:)-ua_est(4,:),'lineWidth',1.2,'color','b');
hold on
plot(tt,0.18*ones(length(tt),1),'lineWidth',1.2,'color','k');
xlabel('Time t(s)','Fontname','Times New Roman','FontSize', 11);
ylabel('Width','Fontname','Times New Roman','FontSize', 11);
title('Widths of prediction intervals','Fontname','Times New Roman','FontSize', 11)
legend({'Widths of prediction intervals for estimating $\dot{p}_{x,1}$ and $\dot{p}_{y,1}$','Widths of prediction intervals for estimating $\dot{p}_{x,2}$ and $\dot{p}_{y,2}$','Width of constant bound'},'Fontname','Times New Roman','FontSize', 11,'Interpreter', 'latex');
axis([0 30 -0.5 1.5])

%=======plot distance====
figure(10)
tt=dt*(0:1:length(x_1(1,:))-1);
d1=sqrt((x_1(1,:)-x_a1(1,:)).^2.+(x_1(2,:)-x_a1(2,:)).^2);
d2=sqrt((x_1(1,:)-x_a2(1,:)).^2.+(x_1(2,:)-x_a2(2,:)).^2);

plot(tt,d1,'r','lineWidth',1.2)
hold on 
plot(tt,d2,'Color',[0.8500 0.3250 0.0980],'lineWidth',1.2)
xlabel('Time t(s)','Fontname','Times New Roman','FontSize', 11);
ylabel('Distance','Fontname','Times New Roman','FontSize', 11);
legend('Distance between ego agent and agent 1','Distance between ego agent and agent 2','Fontname','Times New Roman','FontSize', 11)
axis([0,30,0,3]);

