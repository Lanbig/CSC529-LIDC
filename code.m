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


%% Model Building

%% Decision tree

dt = fitctree(X,Y);
view(dt,'mode','graph')
resubLoss(dt)
loss(dt,X(cv.test,:),Y(cv.test,:))

%% Tree - Bagging
tic
bagTree = fitensemble(X(cv.training,:),Y(cv.training,:),'bag',200,'Tree','Type','classification','nprint',100);
toc

resubLoss(bagTree)
loss(bagTree,X(cv.test,:),Y(cv.test,:))

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