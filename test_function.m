% test function 
clear all
close all
l =[1,2,0.01];
sigman = 1.2;
sigmav = 0.5;
x = linspace(-2,2,120)'+sin(linspace(0,5,120)')+normrnd(0,0.1,120,1);
y =  sin(linspace(0,9,120)')+normrnd(0,0.2,120,1);
z = normrnd(0,0.5,120,1);
X =[x,y,z];
%%  build kernel by for 
for i = 1:120
    for j = 1:120
        K(i,j) = sigman^2*exp(-0.5*sum((X(i,:) - X(j,:)).^2.*(l.^2)));
    end 
end 

%% target value 
target = mvnrnd(zeros(120,1),K,1)' + normrnd(0,sigmav,120,1);

li = abs(normrnd(0,1,3,1));
%% inference from the function 
parfor i = 1:5
[Samples1{i},LMP{i}] = FBGPs(X,target,'psv','halfnormal','numSamples',20,'BI',1000);
end 
%[Samples2] = FBGPs(X,target,log([li;std(target);std(target)]),500,'halfnormal','halfnormal' ,2, 3000);
%% 

a1 = mean(exp(Samples1{1}))
a2 = mean(exp(Samples1{2}))
a3 = mean(exp(Samples1{3}))
a4 = mean(exp(Samples1{4}))
a5 = mean(exp(Samples1{5}))

%% inference from the fitrgp  tool box 
sigma0 = std(target);
sigmaF0 = sigma0;
d = size(X,2);
sigmaM0 = li;

gprMdl = fitrgp(X,target,'Basis','constant','FitMethod','exact',...
'PredictMethod','exact','KernelFunction','ardsquaredexponential',...
'KernelParameters',[sigmaM0;sigmaF0],'Sigma',sigma0,'Standardize',1);

[1./(gprMdl.KernelInformation.KernelParameters(1:end-1));gprMdl.KernelInformation.KernelParameters(end);gprMdl.Sigma]'
sum(abs([1./(gprMdl.KernelInformation.KernelParameters(1:end-1));gprMdl.KernelInformation.KernelParameters(end);gprMdl.Sigma]'-[l,sigman,sigmav]))
