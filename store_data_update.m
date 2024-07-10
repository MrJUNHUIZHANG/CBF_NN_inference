% Update recorded data for concurrent learning

function  [store_data, store_x, store_y]=store_data_update(store_data,store_x,store_y,x_new,y_new,p_max,centers,NN_amount)

eps=0.08; % crition
%[~,N]=size(x);
% for i=1:N-1            % y available
[~,p]=size(store_data);
if p==0
    Phi=RBF(x_new,centers,NN_amount);
    store_data=[store_data Phi];
    store_x=[store_x x_new];
    store_y=[store_y y_new];
elseif p<p_max
    Phi=RBF(x_new,centers,NN_amount);
    if sum((store_data(:,end)-Phi).^2)/sum((store_data(:,end)-Phi).^2)>=eps
        store_data=[store_data Phi];
        store_x=[store_x x_new];
        store_y=[store_y y_new];
    end
elseif p>=p_max
    T=store_data;
    S_old=min(svd(store_data'));
    for j=1:p
        T(:,j)=RBF(x_new,centers,NN_amount);
        S=min(svd(T'));
        if S>S_old
            store_data(:,j)=RBF(x_new,centers,NN_amount);
            store_x(:,j)=x_new;
            store_y(:,j)=y_new;
        end
    end
end

