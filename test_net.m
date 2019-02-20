% --- test network ---
k = 7
res(1,:) = testNet(W_h, W_o, normalize(mnist.test0(k,:)','range'),0);  
res(2,:) = testNet(W_h, W_o, normalize(mnist.test1(k,:)','range'),1);
res(3,:) = testNet(W_h, W_o, normalize(mnist.test2(k,:)','range'),2); 
res(4,:) = testNet(W_h, W_o, normalize(mnist.test3(k,:)','range'),3); 
res(5,:) = testNet(W_h, W_o, normalize(mnist.test4(k,:)','range'),4);   
res(6,:) = testNet(W_h, W_o, normalize(mnist.test5(k,:)','range'),5);
res(7,:) = testNet(W_h, W_o, normalize(mnist.test6(k,:)','range'),6);
res(8,:) = testNet(W_h, W_o, normalize(mnist.test7(k,:)','range'),7);
res(9,:) = testNet(W_h, W_o, normalize(mnist.test8(k,:)','range'),8);
res(10,:) = testNet(W_h, W_o, normalize(mnist.test9(k,:)','range'),9);


function res = testNet(W_h, W_o, input, n)
    b1 = 0.01;
    b2 = 0.01;
% ----- forward propagation -----
    netH = W_h * input + b1*1;
    testH = 1./(1+exp(-netH));    
    netO = W_o * testH + b2 * 1;
    res = 1./(1+exp(-netO));
    
    [acc, ind] = max(res);
    fprintf('Number - %d, prediction - %d, accuracy - %f\n',n, ind-1,acc);
    
    c = 1;
    for i = 1:28
        for j = 1:28
            im(i,j)=input(c);
            c = c + 1;
        end
    end

    %createfigure(im)
end