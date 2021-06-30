  %% Problem formulation:
% Let, error = p - p_target and x_bar = x - x_baseline
% Cost = error'*Q*error + x_bar'*S*x_bar + +\dot(x)'*R*\dot(x)
% System dynamics: \dot(error) = C*velocity; \dot(velocity) = L*velocity + b*f(x)

clear all; 
close all; clc;

%% HORIZON OF THE PROBLEM

global T_end; T_end = 5;
global dt; dt = 0.05; 
global t_steps; t_steps = 0:dt:T_end-dt;

%% DIMENSION OF STATE SPACE AND INPUT SPACE

%dimension of the latent space
global m_dim; m_dim = 2;

%dimension of the neural state space
global n_dim; n_dim = 40; 

%% Import the dynamics of the system
% 
C = [0.9, 0.1; 0.1, 0.9];
% C = eye(m_dim);
friction = [-0.25, 0; 0 -0.25]; %drift in decoder dynamics
overlap_num = 10; %overlap in tuning curves

if overlap_num>n_dim
    error('Total number of neurons must be greater than number of untuned neurons');
end
spread = 10;
b_Matrix = weighting_matrix(n_dim, overlap_num, spread); % tuning matrix

stim_A_num = ceil((overlap_num + n_dim)/ 2);
pure_stim_A = stim_A_num - overlap_num;

% sigmoidal non-linearity
sigma.lower_asymp = -1; sigma.upper_asymp = 100; sigma.curve = 0.05; sigma.mid_pt= 100;

%target position 
target = [1; 0];

%% Set up optimization

iter = 0; max_Iters = 20; 
Error = Inf; %initial error
Tol = 1e-5; 

%load penalty matrices
accuracy = 0.05; energy = 0.005; smoothness = 0.1; 
Q_bar = load_penalty_matrix(accuracy, energy);
R_bar = smoothness*eye(n_dim);

accuracy_f = 20; energy_f = 0.005; %optimization criteria
Q_f = load_penalty_matrix(accuracy_f, energy_f);


if sigma.lower_asymp == 0
    x_baseline = sigma.mid_pt;
else
    x_baseline = sigma.mid_pt - (1/sigma.curve)*log(-sigma.upper_asymp/sigma.lower_asymp);
end

% initialiaze navigational state vector and neural activity
error_init= -target;
velocity_init = zeros(m_dim, 1);
x_init = zeros(n_dim,1);
omega_init = [error_init; velocity_init; x_init];

%initial guess for control
y_vec = 1e-3*randn(n_dim, numel(t_steps));

%forward simulate 
omega_vec = evolve_system(C, friction, b_Matrix, y_vec, omega_init, sigma, x_baseline, target);
omega_perturb_init = zeros(2*m_dim+n_dim,1);

step_size(1) = 1;
color_spec = linspace(0.2,1,max_Iters+1);

while(iter<=max_Iters && Error>Tol)
    
    iter = iter+1;
    %Step 1: Calculate dynamics of the auxiliary problem
    [A_k, B_k] = calculate_auxiliary(omega_vec, y_vec, C, friction, b_Matrix, sigma, x_baseline);
    
    % Step 2a: Compute matrix part of costate vector 
    [T, PHI] = costate_mat(A_k, B_k, Q_bar, Q_f, R_bar);
    % Step 2b: Compute vector part of costate vector
    [t, phi] = costate_vec(A_k, B_k, Q_bar, Q_f, R_bar, PHI, omega_vec, y_vec);
    
    %Step 3: Save current input and augmented state vector
    omega_vec_cache{iter} = omega_vec; 
    omega_vec_2(iter) = norm(omega_vec, 'Fro');
    y_vec_cache{iter} = y_vec;
    
    %Step 4a: Compute state perturbation
    omega_perturb_vec = calc_omega_perturb(omega_perturb_init, A_k, B_k, flipud(PHI), flipud(phi), R_bar, y_vec); 
    omega_perturb_vec_cache{iter} = omega_perturb_vec;
    omega_perturb_vec_2(iter) = norm(omega_perturb_vec, 'Fro');
    
    %Step 4b: Compute cost function
    cost = 0;
    for i = 1:size(omega_vec, 2)-1
        cost = cost+ omega_vec(:,i)'*Q_bar*omega_vec(:,i)+y_vec(:,i)'*R_bar*y_vec(:,i);
    end
    cost = cost+omega_vec(:,end)'*Q_f*omega_vec(:,end);
    cost_per_iter(iter) = cost;
        
    %Step 4c: Compute input perurbation
    y_perturb_vec = calc_y_perturb_vec(B_k, omega_perturb_vec, flipud(PHI), flipud(phi), R_bar, y_vec);
    
    %Step 5a: Update control input
    y_vec = y_vec+step_size(iter)*y_perturb_vec;
    
    %Step 5b: Update state vector
%    omega_vec = evolve_system(C, friction, b_Matrix, y_vec, omega_init, sigma, x_baseline, target);
     omega_vec = omega_vec + omega_perturb_vec;
    
    %Step 6a: Compute error
    Error = norm(omega_vec-omega_vec_cache{iter}, 'Fro');
    omega_diff(iter) = Error; 
        
    %Step 6c: Update step size
    step_size(iter+1) = step_size(iter)*0.99;
    
    pos = target+omega_vec(1:m_dim,:);
    vel = omega_vec(m_dim+1:2*m_dim,:);
    x_response = x_baseline+omega_vec(2*m_dim+1:end,:);
    
%     figure(1)
%     plot(pos(1,:), pos(2,:),'LineWidth',0.5, 'Color', [0,color_spec(iter), 0]); title(num2str(iter-1)); %axis([0 1 0 1]);
%     hold on; plot(0:0.1:1.2, 0:0.1:1.2, '--k','LineWidth',1.5); 
%     axis tight
    
    if iter>=2
       omega_perturb_diff(iter) = norm(omega_perturb_vec_cache{iter} - omega_perturb_vec_cache{iter - 1});
    end
    
    deviation(iter) = norm(pos(:,end)-target);
    if iter>1 && deviation(iter)<0.01
        break
    end
    
   

end
figure(1)
plot(target(1),target(2),'*','Color','k');
hold on
plot(pos(1, 1), pos(2, 1),'o','Color','b'); 
plot(pos(1, end), pos(2, end),'o', 'Color','r'); 
hold off

%% Convergence properties

figure(14),
plot(1:iter, omega_diff, '-s', 'LineWidth', 1.2);
xlim([1, iter]);
% ylabel('||\omega_k - \omega_{k-1}||_{Fro}')
hold on
plot(2:iter, omega_perturb_diff(2:end), '-o', 'LineWidth', 1.2);
% xlim([2, iter]);
% ylabel('||\delta \omega_k - \delta \omega_{k-1}||_{Fro}')

% figure(16),
% plot(1:iter, omega_vec_2, '-o', 'LineWidth', 1.2);
% xlim([2, iter]);
% ylabel('||\omega_k||_{Fro}')

figure(17),
plot(1:iter, omega_perturb_vec_2, '-o', 'LineWidth', 1.2);
xlim([2, iter]);
ylabel('||\delta \omega_k||_{Fro}')

figure(18)
plot(1:iter-1, cost_per_iter(2:end), '-s', 'LineWidth', 1.2);
xlim([1, iter-1])
ylabel('Cost Function')

%% Neural activity

figure(12),
plotshaded(t_steps, x_response(1:pure_stim_A,:),'b');
hold on
plotshaded(t_steps, x_response(pure_stim_A+1:stim_A_num,:),'m');
plotshaded(t_steps, x_response(stim_A_num+1:end,:),'r');
xlim([0, T_end])
hold off

% figure(20)
% polarplot(mean(x_response(1:pure_stim_A,:))-x_baseline,'b', 'LineWidth', 2);
% hold on
% polarplot(mean(x_response(pure_stim_A+1:stim_A_num,:))-x_baseline,'m', 'LineWidth', 2);
% polarplot(mean(x_response(stim_A_num+1:end,:))- x_baseline,'r', 'LineWidth', 2);
% 
figure(22)
% temp = mean(x_response)-x_baseline;
polarhistogram(mean(x_response(1:stim_A_num,:)),10,'FaceColor','blue', 'LineStyle', 'None');
hold on
% polarhistogram(mean(x_response(pure_stim_A+1:stim_A_num,:)),10,'FaceColor','magenta', 'FaceAlpha', 0.5);
polarhistogram(mean(x_response(stim_A_num+1:end,:)),10,'FaceColor','red','LineStyle', 'None');
set(gca, 'RTickLabel', [], 'ThetaTickLabel', []);


%% Relevant functions

function b_Matrix = weighting_matrix(control_num, overlap_num, spread, intensity)

if nargin<4
    intensity = 10;
end

if nargin<3
    spread = 10;
end

stim_A_num = ceil((overlap_num+control_num)/2);
stim_A_mean = ceil(stim_A_num/2); 
stim_B_mean = control_num - stim_A_mean+1;

b_Matrix = intensity*[pdf('Normal',1:control_num, stim_A_mean, spread); pdf('Normal',1:control_num, stim_B_mean, spread)];

end

function Q_bar = load_penalty_matrix(accuracy, energy)

global m_dim; global n_dim;

Q = accuracy*eye(m_dim);
S = energy*eye(n_dim);

Q_bar = blkdiag(Q, zeros(m_dim, m_dim), S);
end

function omega_vec = evolve_system(C, friction, b_Matrix, y_vec, omega_init, sigma, x_baseline, target)

global dt; global t_steps; 
global m_dim; global n_dim;

error_vec = zeros(m_dim, numel(t_steps));
velocity_vec = zeros(m_dim, numel(t_steps));
x_vec = zeros(n_dim, numel(t_steps));

error_vec(:,1) = omega_init(1:m_dim);
velocity_vec(:,1) = omega_init(m_dim+1:2*m_dim);
x_vec(:,1) = omega_init(2*m_dim+1:2*m_dim+n_dim);

for ii = 1:numel(t_steps)-1
    error_vec(:,ii+1) = error_vec(:,ii)+dt*(C*velocity_vec(:,ii));
    velocity_vec(:,ii+1) = velocity_vec(:,ii)+dt*(friction*velocity_vec(:,ii)+b_Matrix*gen_sigmoid(x_baseline+x_vec(:,ii),sigma));
    x_vec(:,ii+1) = x_vec(:,ii)+dt*(y_vec(:,ii));
end

omega_vec = [error_vec; velocity_vec; x_vec];

end

function y = gen_sigmoid(x,sigma)

u = sigma.upper_asymp; l = sigma.lower_asymp; a = sigma.curve; c = sigma.mid_pt;
y = l+((u-l)./(1 + exp(-a.*(x-c))));

end

function [A_k, B_k] = calculate_auxiliary(omega_vec, y_vec, C, friction, b_Matrix, sigma, x_baseline)

global m_dim; global n_dim;
error_vec = omega_vec(1:m_dim,:); velocity_vec = omega_vec(m_dim+1:2*m_dim,:); x_vec = omega_vec(2*m_dim+1:2*m_dim+n_dim,:);
global t_steps;

for t = 1:numel(t_steps)
    A_k_t = [zeros(m_dim), C, zeros(m_dim, n_dim);
             zeros(m_dim), friction, b_Matrix*f_x(x_baseline+x_vec(:,t), sigma);
             zeros(n_dim, 2*m_dim+n_dim)];
    A_k(:,t) = A_k_t(:);
    B_k_t = [zeros(2*m_dim, n_dim); eye(n_dim)];
    B_k(:,t) = B_k_t(:);
end
end
    
function output_mat = f_x(x,sigma)

temp = (gen_sigmoid(x,sigma)-sigma.lower_asymp).*(1-(1/(sigma.upper_asymp-sigma.lower_asymp))*(gen_sigmoid(x,sigma)-sigma.lower_asymp));
output_mat = sigma.curve*diag(temp');
end

function [T, PHI] = costate_mat(A_k, B_k, Q, Q_f, R)

PHI_F = Q_f;
global t_steps;
t_span = fliplr(t_steps);
[T, PHI] = ode45(@(T, PHI)solve_costate_mat(T, PHI, A_k, B_k, Q, R),t_span, PHI_F);
end

function dPHI_dt = solve_costate_mat(t, PHI, A_k, B_k, Q, R)

global m_dim; global n_dim; global t_steps;
total_dim = 2*m_dim+n_dim;
t_span = fliplr(t_steps);

PHI = reshape(PHI, [total_dim, total_dim]);

A_k_t = interp1(t_span, A_k',t);
A_k_t = reshape(A_k_t,[total_dim, total_dim]);

B_k_t = interp1(t_span, B_k',t);
B_k_t = reshape(B_k_t, [total_dim, n_dim]);

dPHI_dt = -(A_k_t'*PHI+PHI*A_k_t-PHI*B_k_t*inv(R)*B_k_t'*PHI+Q);
dPHI_dt = dPHI_dt(:); 

end

function [t, phi] = costate_vec(A_k, B_k, Q, Q_f, R, PHI_K, omega_vec, y_vec)

global n_dim; global t_steps;

phi_F = Q_f*omega_vec(:,end);

t_span = fliplr(t_steps);
[t,phi] = ode45(@(t,phi)solve_costate_vec(t, phi, PHI_K, A_k, B_k, Q, R, y_vec, omega_vec), t_span, phi_F);

end

function dphidt = solve_costate_vec(t, phi, PHI_K, A_k, B_k, Q, R, y_vec, omega_vec)

global m_dim; global n_dim; global t_steps;

total_dim = 2*m_dim+n_dim;
t_span = fliplr(t_steps);

A_k_t = interp1(t_span, A_k', t);
A_k_t = reshape(A_k_t, [total_dim, total_dim]);

B_k_t = interp1(t_span, B_k', t);
B_k_t = reshape(B_k_t, [total_dim, n_dim]);

PHI_t = interp1(t_span, PHI_K, t);
PHI_t = reshape(PHI_t, [total_dim, total_dim]);

y_vec_t = interp1(t_span, y_vec', t);
omega_vec_t = interp1(t_span,omega_vec',t);

dphidt = -(A_k_t' - PHI_t*B_k_t*inv(R)*B_k_t')*phi+PHI_t*B_k_t*y_vec_t'-Q*omega_vec_t';
end

function omega_perturb = calc_omega_perturb(omega_perturb_init, A_k, B_k, PHI, phi, R, y_vec)

global m_dim; global n_dim; global dt; global t_steps;

total_dim = 2*m_dim+n_dim; 
omega_perturb = zeros(total_dim, numel(t_steps));
omega_perturb(:,1) = omega_perturb_init;

for tt = 1:length(t_steps)-1
   A_k_t = reshape(A_k(:,tt+1),[total_dim, total_dim]);
   B_k_t = reshape(B_k(:,tt+1), [total_dim, n_dim]);
   PHI_t = reshape(PHI(tt+1,:),[total_dim, total_dim]);
   phi_t = phi(tt+1,:)';

   omega_perturb(:,tt+1) = omega_perturb(:,tt)+dt*((A_k_t -B_k_t*inv(R)*B_k_t'*PHI_t)*omega_perturb(:,tt)...
                                            -B_k_t*inv(R)*B_k_t'*phi_t - B_k_t*y_vec(:,tt));
end

end

function y_perturb = calc_y_perturb_vec(B_k, omega_perturb_vec, PHI, phi, R, y_vec)

global m_dim; global n_dim; global t_steps;

total_dim = 2*m_dim+n_dim;
y_perturb = zeros(n_dim, numel(t_steps));

for tt = 1:length(t_steps)
    B_k_t = reshape(B_k(:,tt), [total_dim, n_dim]);
    PHI_t = reshape(PHI(tt,:),[total_dim, total_dim]);
    phi_t = phi(tt,:)';
    y_perturb(:,tt) = -inv(R)*B_k_t'*(PHI_t*omega_perturb_vec(:,tt)+phi_t)-y_vec(:,tt);
end
end