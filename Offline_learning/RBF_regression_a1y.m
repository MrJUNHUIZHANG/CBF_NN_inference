% Offline training for RBFNNs (Agent 1 at y direction)
clc
load('u_a1_circle.mat');
load('x_1_circle.mat');
load('x_a1_circle.mat');

load('u_a1_sin.mat');
load('x_1_sin.mat');
load('x_a1_sin.mat');

load('u_a1_spiral.mat');
load('x_1_spiral.mat');
load('x_a1_spiral.mat');
dt=0.01; 

% Generate sample data
P1x = [x_1_circle(1,1:end-1) x_1_sin(1,1:end-1) x_1_spiral(1,1:end-1)];
P1y = [x_1_circle(2,1:end-1) x_1_sin(2,1:end-1) x_1_spiral(2,1:end-1)];
Pax = [x_a1_circle(1,1:end-1) x_a1_sin(1,1:end-1) x_a1_spiral(1,1:end-1)];
Pay = [x_a1_circle(2,1:end-1) x_a1_sin(2,1:end-1) x_a1_spiral(2,1:end-1)];
z1  = [u_a1_circle(1,:) u_a1_sin(1,:) u_a1_spiral(1,:)];
z2 = [u_a1_circle(2,:) u_a1_sin(2,:) u_a1_spiral(2,:)]; 

input = [P1y; Pay];
output = [z2]; 

centers = mean(input, 2);
% choose a spread constant
spread =1;
% choose max number of neurons
K = 8;
% performance goal (SSE)
goal = 0;
% number of neurons to add between displays
Ki = 1;
% create a neural network
net = newrb(input, output, goal, spread, K, Ki);
net_pre=net(input);


W= net.LW{2,1};  % weight
bias = net.b{2}; %  bias

centers = net.IW{1}; % centers
spreads = net.b{1};  % spread  sqrt(log(2))/s


y_math=[];
for i=1:length(input)
    H=RBF(input(:,i),centers,spreads,K);
    y_out=W*H+bias;
    y_math=[y_math y_out];
end

t=dt*(1:1:length(P1x));
plot(t,output(1,:));
hold on
plot(t,net_pre(1,:))
legend('Dynamics','Neural network with offline training')
xlabel('Time');
ylabel('Dynamics of agent 1 at y direction');

W2=W;
centers2=centers;
bias2=bias;
save('W2.mat','W2')
save('centers2.mat','centers2')
save('bias2.mat','bias2')

function y=RBF(x,centers,spreads,K)
H=[];
for i=1:K
    h=exp(-spreads(i)^2*sum((x-centers(i,:)').^2));
    H=[H;h];
end
y=H;
end


