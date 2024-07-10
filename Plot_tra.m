% Plot trajectory for ego and other agents
base_color1 = [0 0 1];
base_color2 = [1 0 0]; 
base_color3 = [0.8500 0.3250 0.0980]; 

x_traj1 = x_1(1,1:10:1500);
y_traj1= x_1(2,1:10:1500);
x_traja1 = x_a1(1,1:10:1500);
y_traja1= x_a1(2,1:10:1500);
x_traja2 = x_a2(1,1:10:1500);
y_traja2= x_a2(2,1:10:1500);

alpha_values= linspace(1,0.1, numel(x_traj1)); 

hold on
for i = 1:numel(x_traj1)
    scatter(x_traj1(i), y_traj1(i), 20, base_color1, 'filled', 'MarkerFaceAlpha', alpha_values(i));
end
hold on
 
for i = 1:numel(x_traja1)
    scatter(x_traja1(i), y_traja1(i), 20, base_color2, 'filled', 'MarkerFaceAlpha', alpha_values(i));
end
hold on

for i = 1:numel(x_traja2)
    scatter(x_traja2(i), y_traja2(i), 20, base_color3, 'filled', 'MarkerFaceAlpha', alpha_values(i));
end
hold on
plot(x_ref(1,:),x_ref(2,:),'k:','lineWidth',0.5)
xlabel('x','Fontname','Times New Roman','FontSize', 11);
ylabel('y','Fontname','Times New Roman','FontSize', 11);
axis([1.5,7,0,3.5]);




