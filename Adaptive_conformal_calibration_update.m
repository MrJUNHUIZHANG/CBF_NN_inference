function [calibration_set,calibration_residual,qn2_k]=Adaptive_conformal_calibration_update(calibration_set,calibration_residual,x_new,y_new, y_est, alpha_t)

[~,b]=size(calibration_set);
residual=abs(y_new-y_est);
n2=300;

%alpha_t;

data_point=[x_new;y_new];

if b<n2   
    calibration_set=[calibration_set data_point];
    calibration_residual=[calibration_residual residual];
else
    calibration_set(:,1)=[];
    calibration_set=[calibration_set data_point];
    calibration_residual(:,1)=[];
    calibration_residual=[calibration_residual residual];
end

[~,b]=size(calibration_set);

[calibration_residual_sort,I]=sort(calibration_residual);

position=ceil((b+1)*(1-alpha_t));

if position>b
   position=b;     %Use previous step as R_imax estimate
end
qn2_k=calibration_residual_sort(position)+0.01;
end