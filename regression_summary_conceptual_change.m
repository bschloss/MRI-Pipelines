pars =              {'201','002','003','004','105','006','107','008','009','110'};
pars = horzcat(pars,{'011','012','013','214','015','016','017','018','019','020'});
pars = horzcat(pars,{'021','122','023','024','025','026','027','028','029','030'});
pars = horzcat(pars,{'031','132','033','034','035','036','037','038','039','040'});
pars = horzcat(pars,{'041','042','043','044','045','046','047','048','049','050'});
context = zeros(length(pars),3);
order = zeros(length(pars),3);
contextorder = zeros(length(pars),3);
sg = zeros(length(pars),3);
bow = zeros(length(pars),3);
sgbow = zeros(length(pars),3);
for i = 1:length(pars)
    par = pars{i};
    load(strcat('/gpfs/group/pul8/default/read/',par,'/fMRI_Analyses/Conceptual_Change/results.mat'));
    context(i,1) = correct.('context')/N;
    context(i,2) = 1-binocdf(correct.('context'),N,.5);
    order(i,1) = correct.('order')/N;
    order(i,2) = 1-binocdf(correct.('order'),N,.5);
    contextorder(i,1) = correct.('contextorder')/N;
    contextorder(i,2) = 1-binocdf(correct.('contextorder'),N,.5);
    sg(i,1) = correct.('sg')/N;
    sg(i,2) = 1-binocdf(correct.('sg'),N,.5);
    bow(i,1) = correct.('bow')/N;
    bow(i,2) = 1-binocdf(correct.('bow'),N,.5);
    sgbow(i,1) = correct.('sgbow')/N;
    sgbow(i,2) = 1-binocdf(correct.('sgbow'),N,.5);
end

context(:,3) = mafdr(context(:,2),'BHFDR','true');
order(:,3) = mafdr(order(:,2),'BHFDR','true');
contextorder(:,3) = mafdr(contextorder(:,2),'BHFDR','true');
sg(:,3) = mafdr(sg(:,2),'BHFDR','true');
bow(:,3) = mafdr(bow(:,2),'BHFDR','true');
sgbow(:,3) = mafdr(sgbow(:,2),'BHFDR','true');
save('/gpfs/group/pul8/default/read/Group_Analyses/Conceptual_Change/results_all_participants.mat','context','order','contextorder','sg','bow','sgbow');
exit;
