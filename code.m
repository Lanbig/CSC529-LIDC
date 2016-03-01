%%
cd ~Lanbig/Dropbox/CSC529/Project
load LIDC


%% Data Preprocesing

% Seperate independent and dependent variables
X = LIDCData(:,9:72);  % independent variables

% Calculate avg of Malignacy value
n = size(LIDCData,1);
malignAvg = zeros(n,1);

for i=1:n
    row = LIDCData(i,:);
    malignAvg(i,1) = round((row(5) + row(6) + row(7) + row(8)) / 4 );   
end

Y = malignAvg; % dependent variables

cv = cvpartition(n, 'holdout', 0.34);

%% Visualize data distribution
figure(1),histogram(Y)
figure(2),histogram(Y(cv.training,:))
figure(3),histogram(Y(cv.test,:))

%% Model Building

%% Decision tree

dt = fitctree(X,Y);
view(dt,'mode','graph')
resubLoss(dt)
loss(dt,X(cv.test,:),Y(cv.test,:))

yHat = predict(dt,X(cv.training,:));
CP = classperf(Y(cv.training,:),yHat);
confusionmat(Y(cv.training),predict(dt,X(cv.training,:)))
confusionmat(Y(cv.test),predict(dt,X(cv.test,:)))


%% Tree - Bagging
tic
bagTree = fitensemble(X(cv.training,:),Y(cv.training,:),'bag',200,'Tree','Type','classification','nprint',100);
toc

resubLoss(bagTree)
loss(bagTree,X(cv.test,:),Y(cv.test,:))

confusionmat(Y(cv.training),predict(bagTree,X(cv.training,:)))
confusionmat(Y(cv.test),predict(bagTree,X(cv.test,:)))

%% Tree - AdaboostM2
tic
adaTree = fitensemble(X(cv.training,:),Y(cv.training,:),'AdaboostM2',1000,'Tree');
toc

resubLoss(adaTree)
loss(adaTree,X(cv.test,:),Y(cv.test,:))

%% Discriminant - Bagging
tic
bagDisc = fitensemble(X(cv.training,:),Y(cv.training,:),'Subspace',1000,'KNN');
toc

resubLoss(bagDisc)
loss(bagDisc,X(cv.test,:),Y(cv.test,:))