% ------------------------------------------------------------------
% sparse network with preconditioned conjugate gradient training procedure
% ----------------||-------------------------|----------------------
% #.Network consists of one hidden layer                            
% #.Network trains on MNIST dataset train examples                   
% #.Sparse network is used                                          
% sparse network initializes at first, and prepares fixed quantity of
% connections, then after each training set zero weighted connections are
% replaced with random other random one and training process continues
% ------------------------------------------------------------------
%   |      Network scheme     |              |
% ------------------------------------------------------------------
%   |  -->(i)-w-(h)-w-(o)-->  |              |
%  | |      \   / \   /      | |             |
%   |        \ /   \ /        |              |
%    |       / \   / \       |               |
%   | |     /   \ /   \     | |              |
%    | -->(i)-w-(h)-w-(o)--> |               |
% ------------------------------------------------------------------
%    |                       |               |
% ------------------------------------------------------------------
clc
clear all;

mnist = load('mnist_all.mat');                 % MNIST dataset

n_input = 784;                                 % input layer nodes
n_hidden = 100;                                % hidden layer nodes
n_output = 10;                                 % output layer nodes

eta = 0.2;                                     % learning rate 0 < eta < 1

W_o = rand(n_output,n_hidden);            % output layer weights
W_h = rand(n_hidden,n_input);             % hidden layer weights

% --- Matrix sparsification, first initialization ---
for i = 1:size(W_o,1)
    for j = 1:size(W_o,2)
        if W_o(i,j) < 0.5
            W_o(i,j) = 0.01*W_o(i,j);
        else
            W_o(i,j) = 0;
        end
    end
end
W_o = sparse(W_o);

for i = 1:size(W_h,1)
    for j = 1:size(W_h,2)
        if W_h(i,j) < 0.5
            W_h(i,j) = 0.01*W_h(i,j);
        else
            W_h(i,j) = 0;
        end
    end
end
W_h = sparse(W_h);

% ----- cycle to train network -----
figure('Name','Output Layer Weights Matrix',...
    'NumberTitle','off','Position', [100 400 1300 200]);

Ntrain = 200;                   % maximum number of training sets 6742
for i = Ntrain:-1:1
    tic
    if i <= 5923 % case 0
        input0 = normalize(mnist.train0(i,:)','range');
        target0 = [1;0;0;0;0;0;0;0;0;0];
        [W_h, W_o]= train(W_h, W_o, input0, target0);
        if i < Ntrain
        %[W_h, W_o] = sparsificate(W_h, W_o);
        end
    end
    if i <= 6742 % case 1
       input1 = normalize(mnist.train1(i,:)','range'); 
       target1 = [0;1;0;0;0;0;0;0;0;0];
       [W_h, W_o]= train(W_h, W_o, input1, target1);
       if i < Ntrain
        %[W_h, W_o] = sparsificate(W_h, W_o);
        end
    end
    if i <= 5958 % case 2
        input2 = normalize(mnist.train2(i,:)','range');
        target2 = [0;0;1;0;0;0;0;0;0;0];
        [W_h, W_o]= train(W_h, W_o, input2, target2);
        if i < Ntrain
        %[W_h, W_o] = sparsificate(W_h, W_o);
        end
    end
    if i <= 6131 % case 3
        input3 = normalize(mnist.train3(i,:)','range');
        target3 = [0;0;0;1;0;0;0;0;0;0];
        [W_h, W_o]= train(W_h, W_o, input3, target3);
        if i < Ntrain
        %[W_h, W_o] = sparsificate(W_h, W_o);
        end
    end
    if i <= 5842 % case 4
        input4 = normalize(mnist.train4(i,:)','range');
        target4 = [0;0;0;0;1;0;0;0;0;0];
        [W_h, W_o]= train(W_h, W_o, input4, target4);
        if i < Ntrain
        %[W_h, W_o] = sparsificate(W_h, W_o);
        end
    end
    if i <= 5421 % case 5
        input5 = normalize(mnist.train5(i,:)','range');
        target5 = [0;0;0;0;0;1;0;0;0;0];
        [W_h, W_o]= train(W_h, W_o, input5, target5);
        if i < Ntrain
        %[W_h, W_o] = sparsificate(W_h, W_o);
        end
    end
    if i <= 5918 % case 6
        input6 = normalize(mnist.train6(i,:)','range');
        target6 = [0;0;0;0;0;0;1;0;0;0];
        [W_h, W_o]= train(W_h, W_o, input6, target6);
        if i < Ntrain
        %[W_h, W_o] = sparsificate(W_h, W_o);
        end
    end
    if i <= 6265 % case 7
        input7 = normalize(mnist.train7(i,:)','range');
        target7 = [0;0;0;0;0;0;0;1;0;0];
        [W_h, W_o]= train(W_h, W_o, input7, target7);
        if i < Ntrain
        %[W_h, W_o] = sparsificate(W_h, W_o);
        end
    end
    if i <= 5851 % case 8
        input8 = normalize(mnist.train8(i,:)','range');
        target8 = [0;0;0;0;0;0;0;0;1;0];
        [W_h, W_o]= train(W_h, W_o, input8, target8);
        if i < Ntrain
        %[W_h, W_o] = sparsificate(W_h, W_o);
        end
    end
    if i <= 5949 % case 9
        input9 = normalize(mnist.train9(i,:)','range');
        target9 = [0;0;0;0;0;0;0;0;0;1];
        [W_h, W_o]= train(W_h, W_o, input9, target9);
        if i < Ntrain
        %[W_h, W_o] = sparsificate(W_h, W_o);
        end
    end
    if i < Ntrain
        [W_h, W_o] = sparsificate(W_h, W_o);
    end
    t(i+1) = toc;
    t(1) = t(1) + t(i+1);
    fprintf('Remaining sets... %d/%d, elapsed time %f seconds\n',i,Ntrain,t(1));
    
    % --- dynamic plot ---
        surfc(W_o);                    % Draw Surface
        view(0,90);
        colormap(flipud(hot));
        colorbar('southoutside');
        drawnow                                     % Draw Plot    
end

figure('Name','Output Layer Weights Matrix','NumberTitle','off')
spy(W_o)
figure('Name','Hidden Layer Weights Matrix','NumberTitle','off')
spy(W_h)
% --- figure --- 
figure('Name','Hidden Layer Weights Matrix Contour Plot','NumberTitle','off')
        contour(W_h);                    
        view(0,90);
        colormap(flipud(hot));
        colorbar('southoutside');

% --- test network ---

res(1,:) = testNet(W_h, W_o, normalize(mnist.test0(1,:)','range'),0);  
res(2,:) = testNet(W_h, W_o, normalize(mnist.test1(1,:)','range'),1);
res(3,:) = testNet(W_h, W_o, normalize(mnist.test2(1,:)','range'),2); 
res(4,:) = testNet(W_h, W_o, normalize(mnist.test3(1,:)','range'),3); 
res(5,:) = testNet(W_h, W_o, normalize(mnist.test4(1,:)','range'),4);   
res(6,:) = testNet(W_h, W_o, normalize(mnist.test5(1,:)','range'),5);
res(7,:) = testNet(W_h, W_o, normalize(mnist.test6(1,:)','range'),6);
res(8,:) = testNet(W_h, W_o, normalize(mnist.test7(1,:)','range'),7);
res(9,:) = testNet(W_h, W_o, normalize(mnist.test8(1,:)','range'),8);
res(10,:) = testNet(W_h, W_o, normalize(mnist.test9(1,:)','range'),9);


function res = testNet(W_h, W_o, input, n)
    b1 = 0.01;
    b2 = 0.01;
% ----- forward propagation -----
    netH = W_h * input + b1 * 1;
    testH = 1./(1+exp(-netH));    
    netO = W_o * testH + b2 * 1;
    res = 1./(1+exp(-netO));
    
    [acc, ind] = max(res);
    fprintf('Number - %d, prediction - %d, accuracy - %f\n',n, ind-1,acc);
    
    c = 1;
    for i = 1:28
        for j = 1:28
            im(i,j) = input(c);
            c = c + 1;
        end
    end
    % --- uncomment below to show tested pictures ---
    %createfigure(im)
end

function [W_h, W_o] = train( W_h, W_o, input, target)
% --- this function trains network with conjugate gradient method ---

% bias values, need to implement update of biases at each iteration of
% backpropagation
netBias = 0.01;
netBias_h = 0.01;

k = 0;         % iteration counter
eta = 0.5;     % learning rate
E = 10;        % mse
eps = 0.001;    % min mse

while E > eps  % stop condition mse = 0.01
    
    % ----- forward propagation -----
    netHiddenLayerValues = W_h * input + netBias_h*1;
    netHiddenLayerOutputs = 1./(1+exp(-netHiddenLayerValues));
    
    netOutputLayerValues = W_o * netHiddenLayerOutputs + netBias * 1;
    netOutputLayerOutputs = 1./(1+exp(-netOutputLayerValues));
    
    % --- calculate mse ---
    E = sum(((target-netOutputLayerOutputs).^2)/2);
    if(E < eps)
        break; % dont update the weights if error is acceptable
    end
    
    % ----- back propagation for output layer -----
    
    % partial derivative of mse error with respect to output layer
    dE_dOutput = -(target - netOutputLayerOutputs);
       
    % partial derivative of output layer outputs with respect to network 
    dOutput_dNetout = netOutputLayerOutputs .* (1 - netOutputLayerOutputs);    
    % partial derivative of network with respect to output layer weights
    dNetout_dw = netHiddenLayerOutputs;    
    % error derivative of output layer
    d_EtotalOut = dE_dOutput .* dOutput_dNetout .* dNetout_dw';
    
    
    % ----- back propagation of error for hidden layer -----
    
    % partial derivative of mse error with respect to hidden layer
    d_EtotalHidden_dOut = sum(dE_dOutput.*dOutput_dNetout.*W_o)';
    % partial derivative of hidden layer outputs with respect to network 
    dOut_dNetHidden = netHiddenLayerOutputs .* (1 - netHiddenLayerOutputs);
    % partial derivative of network with respect to hidden layer weights
    dNetHidden_dw = input;
    % error derivative of hidden layer
    d_EtotalHiddenOut = d_EtotalHidden_dOut .* dOut_dNetHidden .* dNetHidden_dw';
   
    
    % computation of beta coefficient for conjugate addition
    if k == 0
        beta = 0;
        beta_hid = 0;
        p_out = 0;
        p_hid = 0;
    end
    if(k > 0)
        % --- preconditioner ---
% preconditioning need to implement here to make changes in beta parameter
% that updates conjugate direction
        
                    % Polak–Ribiere formula
        beta = (d_EtotalOut .* (d_EtotalOut-gradE_prev))...
             ./(gradE_prev .* gradE_prev + 0.0001);
        beta_hid = (d_EtotalHiddenOut .* (d_EtotalHiddenOut-gradE_prev_hid))...
                 ./(gradE_prev_hid .* gradE_prev_hid + 0.0001);        
    end
    
    gradE_prev = d_EtotalOut;
    gradE_prev_hid = d_EtotalHiddenOut; 
    
    % --- conjugate direction calculation ---
    p_out = d_EtotalOut + beta .* p_out;
    p_hid = d_EtotalHiddenOut + beta_hid .* p_hid;
    
    % --- weights update ---
    W_h = W_h - eta*p_hid;
    W_o = W_o - eta*p_out;

    
    k = k + 1;  % iterator to compute error for output curve
    Etotal(k) = E;   % total error for each iteration
    
    % code below is for cycles restriction in case of nonconvergence 
    if(k>5000)
        E = 0.000001;
    end
    %eta  = double(1/sqrt(k));
    
end
end

function [W_h, W_o] = sparsificate(W_h, W_o)
%--- Matrix sparsification function---

n_s = 0;
for i = 1:size(W_o,1)
    for j = 1:size(W_o,2)
        if abs(W_o(i,j)) > 0.05
            W_o(i,j) = W_o(i,j);
        else
            W_o(i,j) = 0;
            n_s = n_s + 1;
        end
        if n_s == 1
            if j <= size(W_o,2) - 20
                for k = 1:20
                    if W_o(i,j+k) == 0
                        W_o(i,j+k) = 0.01*rand;
                        break;
                    end
                end
            else
                for k = 1:20
                    if W_o(i,j-k) == 0
                        W_o(i,j-k) = 0.01*rand;
                        break;
                    end
                end
            end
        end
    end
end
W_o = sparse(W_o);

n_s = 0;
for i = 1:size(W_h,1)
    for j = 1:size(W_h,2)
        if abs(W_h(i,j)) > 0.02
            W_h(i,j) = W_h(i,j);
        else
            W_h(i,j) = 0;
            n_s = n_s + 1;
        end
        if n_s == 1
            if j <= size(W_o,2) - 20
                for k = 1:20
                    if W_h(i,j+k) == 0
                        W_h(i,j+k) = 0.01*rand;
                        break;
                    end
                end
            else
                for k = 1:20
                    if W_h(i,j-k) == 0
                        W_h(i,j-k) = 0.01*rand;
                        break;
                    end
                end
            end
        end
    end
end
W_h = sparse(W_h);
end

