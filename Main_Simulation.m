clc
clear
%=======================================================================
% Load initial values for weight, centers obtained from offline learning
load('W1.mat');
load('bias1.mat');
load('centers1.mat');
load('W2.mat');
load('bias2.mat');
load('centers2.mat');
load('W3.mat');
load('bias3.mat');
load('centers3.mat');
load('W4.mat');
load('bias4.mat');
load('centers4.mat');
W1=[W1 bias1]';
W2=[W2 bias2]';
W3=[W3 bias3]';
W4=[W4 bias4]';
centers1=centers1';
centers2=centers2';
centers3=centers3';
centers4=centers4';

W1_online=zeros(9,1);
W2_online=zeros(9,1);
W3_online=zeros(9,1);
W4_online=zeros(9,1);

%=================================================
% normal agents

x_1=[2.5 2 -pi/2]';           % initial position of ego agent
x_ref=[2.5 2 -pi/2]'; 
v1=1.15;                      % parameters in reference control
u_1=[];                       % real control for ego agent

%-------------------------------------------------
% CBF paramters
d_safe=0.15;                  % safety distance
alpha=1;                      % linear k-calss function 
belta=1.01;                   % paramters in cbf
phi=0.05;

% Two other agents
x_a1=[3 2.8]';                 % initial position of other agents
x_a2=[3.6 2.6]';  
gamma_1=0.5;                   % speed
gamma_2=0.3;
u_a1=[];                       % The actions for agent1
u_a2=[];                       % The actions for agent2

%========================================================================
dt=0.01;                            % Time step
N=3001;                             % The amount total time step

NN_amount=[8,8,8,8];                % The amount of neurons

%==========  Initial online ========

ua_est=[];
ua_offline=ua_est;   % 4*N
ua_online=ua_est;    % 4*N
ua_ub=ua_est;        % 4*N
ua_lb=ua_est;        % 4*N
qn2_k1=0; qn2_k2=0; qn2_k3=0; qn2_k4=0;


% =========Paramters in the stored data for adaptive law

index=[0 0 0 0];       % Check if enough indepent columns are collected 
p_max=[50 50 50 50];   % The maximum amount of collected data
  

%==========initial stored data for adaptive law
store_data1=[];  store_data2=[]; store_data3=[];store_data4=[];  
store_x1=[];  store_x2=[];  store_x3=[];  store_x4=[];  %x
store_y1=[];  store_y2=[];  store_y3=[];  store_y4=[];  %ua

%==========initial values for adaptive conformal prediction
calibration_set1=[]; calibration_set2=[]; calibration_set3=[]; calibration_set4=[];
calibration_residual1=[]; calibration_residual2=[]; calibration_residual3=[]; calibration_residual4=[];
err=zeros(4,N-1); alpha_tar=0.01; alpha_ada=alpha_tar*ones(4,1); gamma=0.002;

%========================Simulation process================================
for i=1:1:N-1
    % Current state 
    x_cur=x_1(:,end);
    xa1_cur=x_a1(:,end);
    xa2_cur=x_a2(:,end);
    if i>1
        p1=[x_1(1,end-1);x_a1(1,end-1)];      % Input consists of states of ego and other agents
        p2=[x_1(2,end-1);x_a1(2,end-1)];
        p3=[x_1(1,end-1);x_a1(1,end-1);x_a2(1,end-1)];
        p4=[x_1(2,end-1);x_a1(2,end-1);x_a2(2,end-1)];
        
        %Update recorded data for concurrent learning
        [store_data1,store_x1,store_y1]=store_data_update(store_data1,store_x1,store_y1,p1,u_a1(1,end),p_max(1),centers1,NN_amount(1));
        [store_data2,store_x2,store_y2]=store_data_update(store_data2,store_x2,store_y2,p2,u_a1(2,end),p_max(2),centers2,NN_amount(2));
        [store_data3,store_x3,store_y3]=store_data_update(store_data3,store_x3,store_y3,p3,u_a2(1,end),p_max(3),centers3,NN_amount(3));
        [store_data4,store_x4,store_y4]=store_data_update(store_data4,store_x4,store_y4,p4,u_a2(2,end),p_max(4),centers4,NN_amount(4));
        
        % adaptive conformal prediction (update adaptive failure probalibility)
        alpha_ada1_next=alpha_ada(1,end)+gamma*(alpha_tar-err(1,end));
        alpha_ada2_next=alpha_ada(2,end)+gamma*(alpha_tar-err(2,end));
        alpha_ada3_next=alpha_ada(3,end)+gamma*(alpha_tar-err(3,end));
        alpha_ada4_next=alpha_ada(4,end)+gamma*(alpha_tar-err(4,end));
        alpha_ada_next=[alpha_ada1_next,alpha_ada2_next,alpha_ada3_next,alpha_ada4_next]';
        alpha_ada=[alpha_ada alpha_ada_next];
        
        % Update calibration dataset for adaptive conformal
        % prediction/calculate widths of prediction sets
        [calibration_set1,calibration_residual1,qn2_k1]=Adaptive_conformal_calibration_update(calibration_set1,calibration_residual1,p1,u_a1(1,end),ua_est(1,end),alpha_ada(1,end));
        [calibration_set2,calibration_residual2,qn2_k2]=Adaptive_conformal_calibration_update(calibration_set2,calibration_residual2,p2,u_a1(2,end),ua_est(2,end),alpha_ada(2,end));
        [calibration_set3,calibration_residual3,qn2_k3]=Adaptive_conformal_calibration_update(calibration_set3,calibration_residual3,p3,u_a2(1,end),ua_est(3,end),alpha_ada(3,end));
        [calibration_set4,calibration_residual4,qn2_k4]=Adaptive_conformal_calibration_update(calibration_set4,calibration_residual4,p4,u_a2(2,end),ua_est(4,end),alpha_ada(4,end));
        
        C12=max(qn2_k1,qn2_k2);C34=max(qn2_k3,qn2_k4);
        qn2_k1=C12; qn2_k2=C12; qn2_k3=C34; qn2_k4=C34; 
        
        % Check if enough data are collected for concurrent learning
        index(1)=check_rank(store_data1,NN_amount(1)+1);
        index(2)=check_rank(store_data2,NN_amount(2)+1);
        index(3)=check_rank(store_data3,NN_amount(3)+1);
        index(4)=check_rank(store_data4,NN_amount(4)+1);

        %---online update weights of RBFNNs--
        W1_online_out=w_ada_exp(W1_online(:,end),u_a1(1,end),p1,store_x1,store_y1,index(1),centers1,NN_amount(1),dt);
        W2_online_out=w_ada_exp(W2_online(:,end),u_a1(2,end),p2,store_x2,store_y2,index(2),centers2,NN_amount(2),dt);
        W3_online_out=w_ada_exp(W3_online(:,end),u_a2(1,end),p3,store_x3,store_y3,index(3),centers3,NN_amount(3),dt);
        W4_online_out=w_ada_exp(W4_online(:,end),u_a2(2,end),p4,store_x4,store_y4,index(4),centers4,NN_amount(4),dt);
        W1_online=[W1_online W1_online_out];
        W2_online=[W2_online W2_online_out];
        W3_online=[W3_online W3_online_out];
        W4_online=[W4_online W4_online_out];
    else

    W1_out=W1;
    W2_out=W2;
    W3_out=W3;
    W4_out=W4;

    W1_online_out=W1_online;
    W2_online_out=W2_online;
    W3_online_out=W3_online;
    W4_online_out=W4_online;

    end

    p1=[x_1(1,end);x_a1(1,end)];               
    p2=[x_1(2,end);x_a1(2,end)];
    p3=[x_1(1,end);x_a1(1,end);x_a2(1,end)];
    p4=[x_1(2,end);x_a1(2,end);x_a2(2,end)];

    ua_est_out=[W1_out'*RBF(p1,centers1,NN_amount(1));
        W2_out'*RBF(p2,centers2,NN_amount(2));
        W3_out'*RBF(p3,centers3,NN_amount(3));
        W4_out'*RBF(p4,centers4,NN_amount(4))];      % 4*N

    ua_online_out=[W1_online_out'*RBF(p1,centers1,NN_amount(1));
        W2_online_out'*RBF(p2,centers2,NN_amount(2));
        W3_online_out'*RBF(p3,centers3,NN_amount(3));
        W4_online_out'*RBF(p4,centers4,NN_amount(4))];      % 4*N
    qnk=[qn2_k1;qn2_k2;qn2_k3;qn2_k4];
    
    ua_ub_out=ua_est_out+qnk;
    ua_lb_out=ua_est_out-qnk;
    ua_offline_out=[W1(:,1)'*RBF(p1,centers1,NN_amount(1));
        W2(:,1)'*RBF(p2,centers2,NN_amount(2));
        W3(:,1)'*RBF(p3,centers3,NN_amount(3));
        W4(:,1)'*RBF(p4,centers4,NN_amount(4))];
 
    ua_est=[ua_est ua_est_out];
    ua_ub=[ua_ub ua_ub_out];
    ua_lb=[ua_lb ua_lb_out];
    ua_online=[ua_online ua_online_out];
    ua_offline=[ua_offline ua_offline_out];


   %=========================================================
    % calculate control for other agents 
    vec1=[x_cur(1)-xa1_cur(1) x_cur(2)-xa1_cur(2)]';                             % alpha h/alpha xa
    ua1_cur=gamma_1*(sqrt(sum(vec1.^2)))*vec1./(sqrt(sum(vec1.^2)));             % action for agent 1     
    vec2=[1*(x_cur(1)-xa2_cur(1))-0.6*(xa1_cur(1)-xa2_cur(1)) 1*(x_cur(2)-xa2_cur(2))-0.6*(xa1_cur(2)-xa2_cur(2))]'; 
    ua2_cur=gamma_2*vec2;                                                        % action for agent 2
    xa1_next=runge_kutta4si(xa1_cur, ua1_cur, dt);       % next state
    xa2_next=runge_kutta4si(xa2_cur, ua2_cur, dt);       % next state
     
    u_a1=[u_a1 ua1_cur];                                 % Save actions and states
    u_a2=[u_a2 ua2_cur]; 
    x_a1=[x_a1 xa1_next];
    x_a2=[x_a2 xa2_next];


 % Check miscoverage

  if u_a1(1,i)<ua_lb(1,i)|| u_a1(1,i)>ua_ub(1,i)
      err(1,i)=1;
  end
  if u_a1(2,i)<ua_lb(2,i)|| u_a1(2,i)>ua_ub(2,i)
      err(2,i)=1;
  end
  if u_a2(1,i)<ua_lb(3,i)|| u_a2(1,i)>ua_ub(3,i)
      err(3,i)=1;
  end
  if u_a2(2,i)<ua_lb(4,i)|| u_a2(2,i)>ua_ub(4,i)
      err(4,i)=1;
  end

 %==============================================================
 %  Reference control for ego agent
    t=dt*i;
    if i<314
         u_ref=[0.6 1];
    elseif i<314*3
         u_ref=[0.6 -1];
    else
        u_ref=[0.6 1];
    end

    x_ref_next=runge_kutta4uni(x_ref(:,end), u_ref, dt);
    x_ref=[x_ref x_ref_next];                             % Reference trajectory of ego

    % CBF-based safe controller with prediction sets
    names = {'u_1', 'u_2'};
    model.varnames = names;
    model.Q = sparse([1 0; 0 1]);
    
    theta=x_cur(3);
    s1=sin(theta)*(x_cur(1)-xa1_cur(1))+cos(theta)*(x_cur(2)-xa1_cur(2));
    h1=sum((x_cur(1:2)-xa1_cur(1:2)).^2)-belta*d_safe^2-sigma(s1);
    dh1_dxi=[2*(x_cur(1)-xa1_cur(1))-sigma_der(s1)*sin(theta),2*(x_cur(2)-xa1_cur(2))-sigma_der(s1)*cos(theta), ...
        -sigma_der(s1)*(cos(theta)*(x_cur(1)-xa1_cur(1))-sin(theta)*(x_cur(2)-xa1_cur(2)))];
    dh1_dxa1=[-2*(x_cur(1)-xa1_cur(1))+sigma_der(s1)*sin(theta) -2*(x_cur(2)-xa1_cur(2))+sigma_der(s1)*sin(theta)];
    g=[cos(theta) 0;sin(theta) 0;0 1];
    dh1_dxig=dh1_dxi*g;   %Lh1*g

    lb=ua_lb_out(1:2)';  % [lb,ub] is the prediction interval
    ub=ua_ub_out(1:2)';
    f1=dh1_dxa1;

    [x,fav1] = linprog(f1,[],[],[],[],lb,ub); 
    
    s2=sin(theta)*(x_cur(1)-xa2_cur(1))+cos(theta)*(x_cur(2)-xa2_cur(2));
    h2=sum((x_cur(1:2)-xa2_cur(1:2)).^2)-belta*d_safe^2-sigma(s2);
    dh2_dxi=[2*(x_cur(1)-xa2_cur(1))-sigma_der(s2)*sin(theta),2*(x_cur(2)-xa2_cur(2))-sigma_der(s2)*cos(theta), ...
        -sigma_der(s2)*(cos(theta)*(x_cur(1)-xa2_cur(1))-sin(theta)*(x_cur(2)-xa2_cur(2)))];
    dh2_dxa2=[-2*(x_cur(1)-xa2_cur(1))+sigma_der(s2)*sin(theta) -2*(x_cur(2)-xa2_cur(2))+sigma_der(s2)*sin(theta)];
    dh2_dxig=dh2_dxi*g;

    lb=ua_lb_out(3:4)';
    ub=ua_ub_out(3:4)';
    f2=dh2_dxa2;

    [x,fav2] = linprog(f2,[],[],[],[],lb,ub);
    
    % Solve CBF-based optimization problem

    model.A = sparse([dh1_dxig;dh2_dxig]);
    model.rhs = [-alpha*h1-fav1+phi;-alpha*h2-fav2+phi];
    model.lb = [-100 -100];
    model.ub = [100 100];
    model.obj = -2*u_ref';
    model.sense = '>';
    model.vtype = 'C';
    gurobi_write(model, 'qp.lp');
    results = gurobi(model);
    u_cur=results.x;
    x_next=runge_kutta4uni(x_cur, u_cur, dt);
    x_1=[x_1 x_next];
    u_1=[u_1 u_cur];

   % Check collision
    if sum((x_cur(1:2)-xa1_cur(1:2)).^2)<d_safe^2||sum((x_cur(1:2)-xa2_cur(1:2)).^2)<d_safe^2
        disp('Collision')
        break
    end
end

%=================plot trajectory==========================
plot(x_1(1,:),x_1(2,:),'b')
hold on 
plot(x_a1(1,:),x_a1(2,:),'r')
hold on 
plot(x_a2(1,:),x_a2(2,:),'Color',[0.8500 0.3250 0.0980])
axis([2,8,0,5]);
figure;
line1 = animatedline('Color', 'b');
line2 = animatedline('Color', 'r');
line3 = animatedline('Color', [0.8500 0.3250 0.0980]);
axis([2,8,0,5]);
xlabel('x');
ylabel('y');
legend('Ego Agent', 'Agents 1','Agents 2');
title('Trajectory');

for i = 1:length(x_1(1,:))
    addpoints(line1, x_1(1,i), x_1(2,i));
    addpoints(line2, x_a1(1,i), x_a1(2,i));
    addpoints(line3, x_a2(1,i), x_a2(2,i));
    drawnow;
    pause(0.01);
end

save('u_1.mat','u_1')
save('x_1.mat','x_1')
save('u_a1.mat','u_a1')
save('x_a1.mat','x_a1')
save('u_a2.mat','u_a2')
save('x_a2.mat','x_a2')

%====================================================
%Functions for dynamics

function xdot= f(x, u) % Unicycle 
xdot=[u(1)*cos(x(3));
     u(1)*sin(x(3));
     u(2)];
end

function x_next = runge_kutta4uni(x, u, dt)
%Runge-Kutta 4 integration
k1 = f(x,         u);
k2 = f(x+dt/2*k1, u);
k3 = f(x+dt/2*k2, u);
k4 = f(x+dt*k3,   u);
x_next = x + dt/6*(k1+2*k2+2*k3+k4);
end

function xdot= fsi(x, u) % Single intergter 
xdot=[u(1);
     u(2)];
end

function x_next = runge_kutta4si(x, u, dt)
% Runge-Kutta 4 integration
k1 = fsi(x,         u);
k2 = fsi(x+dt/2*k1, u);
k3 = fsi(x+dt/2*k2, u);
k4 = fsi(x+dt*k3,   u);
x_next = x + dt/6*(k1+2*k2+2*k3+k4);
end

%====================================================
% Define RBF basis function
function y=RBF(x,centers,K)
spread=0.8326;
H=[];
for i=1:K
    h=exp(-spread^2*sum((x-centers(:,i)).^2));
    H=[H;h];
end
y=[H;1];
end

% Update law with concurent learning convergence
function wdot= f_exp(W,y,x,store_x,store_y,index,centers,K)
tau=1*eye(K+1);

v=W'*RBF(x,centers,K);
epsional_x=v-y;

[~,p]=size(store_x);
sec_term=zeros(K+1,1);

for i=1:p
    epsional=W'*RBF(store_x(:,i),centers,K)-store_y(i);
    sec_term=sec_term-2*RBF(store_x(:,i),centers,K)*epsional';
end
if index==1
    wdot=-tau*RBF(x,centers,K)*epsional_x'+0.01*sec_term;
else
    wdot=-tau*RBF(x,centers,K)*epsional_x';
end
end

% 
function x_next =w_ada_exp(w,y,x,store_x,store_y,index,centers,K,dt)
% Runge-Kutta 4 integration
k1 = f_exp(w,y,x,store_x,store_y,index,centers,K);
k2 = f_exp(w+dt/2*k1,y,x,store_x,store_y,index,centers,K);
k3 = f_exp(w+dt/2*k2,y,x,store_x,store_y,index,centers,K);
k4 = f_exp(w+dt*k3,y,x,store_x,store_y,index,centers,K);
x_next = w + dt/6*(k1+2*k2+2*k3+k4);
end

function y=sigma(s)
k1=10;
y=(exp(k1-s)-1)/(exp(k1-s)+1);
end

function y=sigma_der(s)
k1=10;
y=-2*exp(k1-s)/((exp(k1-s)+1)^2);
end




