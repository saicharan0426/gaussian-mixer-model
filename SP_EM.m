clc;
close all;
%% Q1- Generation of Dataset
N=500;                      % total number of samples
p_1=0.8;                  
m_1=[3,3];                  % Mean vector for cluster 1
sigma_1=[1,0; 0,2];         % Covariance matrix for cluster 1
cluster_1=mvnrnd(m_1,sigma_1,p_1*N);   
p_2=0.2;                   
m_2=[1,-3];                 % mean vector for cluster 2
sigma_2=[2,0; 0,1];         % covariance matrix for cluster 2
cluster_2=mvnrnd(m_2,sigma_2,p_2*N);
Data=[cluster_1;cluster_2]; % Total data
figure("Name","Dataset")
scatter(Data(:,1), Data(:,2),"black",".")
title("Dataset")
%% Q2- EM Algorithm
mu_1=[0,0];                 % random intitialization of parameters
mu_2=[1,1];
pi_1=0.3;
pi_2=0.7;
sigma_1=diag([1,2]);
sigma_2=diag([1,1]);
q_n1=0.25*ones(N,1); 
q_n2=0.75*ones(N,1);        % note- q_n1+q_n2 = 1
iteration=30;               % Total number of iterations
Bound=zeros(1,iteration);
Likelihood = zeros(1,iteration);
for j=1:iteration
pi1_update=mean(q_n1);                                     % new pi_1
mu1_update=sum(q_n1.*Data)/sum(q_n1);                      % new u_1
sigma1_update=(q_n1.*(Data-mu_1))'*(Data-mu_1)/sum(q_n1);  % new sigma_1
pi2_update=mean(q_n2);                                     % new pi_2
mu2_update=sum(q_n2.*Data)/sum(q_n2);                      % new mu_2
sigma2_update=(q_n2.*(Data-mu_2))'*(Data-mu_2)/sum(q_n2);  % new sigma_2
p_xn_1=zeros(N,1);
p_xn_2=zeros(N,1);
    for i =1:N
        p_xn_1(i)=pi1_update/(2*pi*sqrt(det(sigma1_update)))*exp(-0.5*(Data(i,:)-mu1_update)*pinv(sigma1_update)*(Data(i,:)-mu1_update)'); 
        p_xn_2(i)=pi2_update/(2*pi*sqrt(det(sigma2_update))) * exp(-0.5*(Data(i,:)-mu2_update)*pinv(sigma2_update)*(Data(i,:)-mu2_update)');  
    end
q_n1=p_xn_1./(p_xn_1+p_xn_2);                          % new q_n1
q_n2=p_xn_2./(p_xn_1+p_xn_2);                          % new q_n2
pi_1=pi1_update;                                       % updating pi_1
mu_1=mu1_update;                                       % Updating mu_1
sigma_1=sigma1_update;                                 % Updating sigma_1
pi2=pi2_update;                                        % Updating pi_2
mu_2=mu2_update;                                       % Updating mu_2
sigma_2=sigma2_update;                                 % Updating sigma_2
Bound(j)=sum(q_n1*log(pi_1)+q_n2*log(pi2))+sum(q_n1.*(log(p_xn_1/pi_1))+ q_n2.*(log(p_xn_2/pi2)))-sum(q_n1.*log(q_n1)+ q_n2.*log(q_n2)); % lower bound of likelihood
Likelihood(j)=sum(log(p_xn_1+p_xn_2));                 % likelihood
end
%% Plot of Likelihood and Bound 
figure;
plot(Bound);              % plot of Lower Bound
hold on;
plot(Likelihood,'--');    % plot of Likelihood
xlabel('Iterations');
ylabel('Lower Bound');
title('Lower Bound & Likelihood');
legend('Lower Bound','Likelihood');
grid;
%% Q3- Assignment of dataset points to clusters
y=-10:0.1:10;           
x=-4:0.1:10;            
[X1,X2]=meshgrid(x,y);           %2D grid
X=[X1(:) X2(:)];
P=mvnpdf(X,mu_1,sigma_1);        % Multivariate Gaussian pdf(mu_1,sigma_1)
P=reshape(P,length(y),length(x));
figure;
contour(x,y,P);                  % Its contour plot
hold on;
xlabel('x1');
ylabel('x2');
Z=mvnpdf(X,mu_2,sigma_2);        % Multivariate Gaussian pdf(mu_2,sigma_2)
Z=reshape(Z,length(y),length(x));
contour(x,y,Z);                  % Its contour plot
title('Clustered Dataset');
plot(cluster_1(:,1),cluster_1(:,2),'.');   % Plotting the data points in cluster 1
hold on;
plot(cluster_2(:,1),cluster_2(:,2),'.');   % Plotting the data points in cluster 2
grid;