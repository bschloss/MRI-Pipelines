% The following code executes a simulation in which all possible 2 word combinations are left out, 
% a model is built from the remaining nouns, and the brain activation for the remaining two are predicted 
% from the model. Then a test is performed in which the model guesses which of the two left out vectors 
% corresponds to which of the two left out words by comparing the cosine similarities of the predicted and actual vectors. 

if length(num2str(par)) == 1
    par = strcat('00',num2str(par));
elseif length(num2str(par)) == 2
    par = strcat('0',num2str(par));
else
    par = num2str(par);
end
context_designs = strcat('/gpfs/group/pul8/default/read/',par,'/fMRI_Analyses/Conceptual_Change/design_context_svd.mat');
load(context_designs);
order_designs = strcat('/gpfs/group/pul8/default/read/',par,'/fMRI_Analyses/Conceptual_Change/design_order_svd.mat');
load(order_designs);
contextorder_designs = strcat('/gpfs/group/pul8/default/read/',par,'/fMRI_Analyses/Conceptual_Change/design_contextorder_svd.mat');
load(contextorder_designs);
sg_designs = strcat('/gpfs/group/pul8/default/read/',par,'/fMRI_Analyses/Conceptual_Change/design_sg_svd.mat');
load(sg_designs);
bow_designs = strcat('/gpfs/group/pul8/default/read/',par,'/fMRI_Analyses/Conceptual_Change/design_bow_svd.mat');
load(bow_designs);
sgbow_designs = strcat('/gpfs/group/pul8/default/read/',par,'/fMRI_Analyses/Conceptual_Change/design_sgbow_svd.mat');
load(sgbow_designs);
responses = strcat('/gpfs/group/pul8/default/read/',par,'/fMRI_Analyses/Conceptual_Change/response2_450.mat');
load(responses);
context_lo = strcat('/gpfs/group/pul8/default/read/',par,'/fMRI_Analyses/Conceptual_Change/lo_context_svd.mat');
load(context_lo);
order_lo = strcat('/gpfs/group/pul8/default/read/',par,'/fMRI_Analyses/Conceptual_Change/lo_order_svd.mat');
load(order_lo);
contextorder_lo = strcat('/gpfs/group/pul8/default/read/',par,'/fMRI_Analyses/Conceptual_Change/lo_contextorder_svd.mat');
load(contextorder_lo);
sg_lo = strcat('/gpfs/group/pul8/default/read/',par,'/fMRI_Analyses/Conceptual_Change/lo_sg_svd.mat');
load(sg_lo);
bow_lo = strcat('/gpfs/group/pul8/default/read/',par,'/fMRI_Analyses/Conceptual_Change/lo_bow_svd.mat');
load(bow_lo);
sgbow_lo = strcat('/gpfs/group/pul8/default/read/',par,'/fMRI_Analyses/Conceptual_Change/lo_sgbow_svd.mat');
load(sgbow_lo);
response_lo = strcat('/gpfs/group/pul8/default/read/',par,'/fMRI_Analyses/Conceptual_Change/lo_response2_450.mat');
load(response_lo);

N = size(response2_450, 1); 
ncomp = 25;
correct = struct();
correct.('context') = 0;
correct.('order') = 0;
correct.('contextorder') = 0;
correct.('sg') = 0;
correct.('bow') = 0;
correct.('sgbow') = 0;
correct.('contextsgbow') = 0;
correct.('contextordersgbow') = 0;

for i = 1:N

    %initialize matrices for linear regression
    x_context = squeeze(zscore(design_context_svd(i,:,:)));
    x_order = squeeze(zscore(design_order_svd(i,:,:)));
    x_contextorder = squeeze(zscore(design_contextorder_svd(i,:,:)));
    x_sg = squeeze(zscore(design_sg_svd(i,:,:)));
    x_bow = squeeze(zscore(design_bow_svd(i,:,:)));
    x_sgbow = squeeze(zscore(design_sgbow_svd(i,:,:)));
    x_contextsgbow = horzcat(x_context,x_sg,x_bow);
    x_contextordersgbow = horzcat(x_context,x_order,x_sg,x_bow);
    y_matrix = squeeze(zscore(response2_450(i,:, :)));

    %run a regression for each of the 500 selected voxels with each x matrix

    [XL_context,YL_context,XS_context,YS_context,BETA_context,PCTVAR_context] = plsregress(y_matrix,x_context,ncomp);
    [XL_order,YL_order,XS_order,YS_order,BETA_order,PCTVAR_order] = plsregress(y_matrix,x_order,ncomp);
    [XL_contextorder,YL_contextorder,XS_contextorder,YS_contextorder,BETA_contextorder,PCTVAR_contextorder] = plsregress(y_matrix,x_contextorder,ncomp);
    [XL_sg,YL_sg,XS_sg,YS_sg,BETA_sg,PCTVAR_sg] = plsregress(y_matrix,x_sg,ncomp);
    [XL_bow,YL_bow,XS_bow,YS_bow,BETA_bow,PCTVAR_bow] = plsregress(y_matrix,x_bow,ncomp);
    [XL_sgbow,YL_sgbow,XS_sgbow,YS_sgbow,BETA_sgbow,PCTVAR_sgbow] = plsregress(y_matrix,x_sgbow,ncomp);
    [XL_contextsgbow,YL_contextsgbow,XS_contextsgbow,YS_contextsgbow,BETA_contextsgbow,PCTVAR_contextsgbow] = plsregress(y_matrix,x_contextsgbow,ncomp);
    [XL_contextordersgbow,YL_contextordersgbow,XS_contextordersgbow,YS_contextordersgbow,BETA_contextordersgbow,PCTVAR_contextordersgbow] = plsregress(y_matrix,x_contextordersgbow,ncomp);
    
    %pctvar = pctvar + PCTVAR;
    
    %get the values of the semantic features that will be used to predict activation for word1 and word2

    context_pred1 = [1 squeeze(zscore(lo_response2_450(i,1,:)))']*BETA_context;
    context_pred2 = [1 squeeze(zscore(lo_response2_450(i,2,:)))']*BETA_context;
    order_pred1 = [1 squeeze(zscore(lo_response2_450(i,1,:)))']*BETA_order;
    order_pred2 = [1 squeeze(zscore(lo_response2_450(i,2,:)))']*BETA_order;
    contextorder_pred1 = [1 squeeze(zscore(lo_response2_450(i,1,:)))']*BETA_contextorder;
    contextorder_pred2 = [1 squeeze(zscore(lo_response2_450(i,2,:)))']*BETA_contextorder;
    sg_pred1 = [1 squeeze(zscore(lo_response2_450(i,1,:)))']*BETA_sg;
    sg_pred2 = [1 squeeze(zscore(lo_response2_450(i,2,:)))']*BETA_sg;
    bow_pred1 = [1 squeeze(zscore(lo_response2_450(i,1,:)))']*BETA_bow;
    bow_pred2 = [1 squeeze(zscore(lo_response2_450(i,2,:)))']*BETA_bow;
    sgbow_pred1 = [1 squeeze(zscore(lo_response2_450(i,1,:)))']*BETA_sgbow;
    sgbow_pred2 = [1 squeeze(zscore(lo_response2_450(i,2,:)))']*BETA_sgbow;
    contextsgbow_pred1 = [1 squeeze(zscore(lo_response2_450(i,1,:)))']*BETA_contextsgbow;
    contextsgbow_pred2 = [1 squeeze(zscore(lo_response2_450(i,2,:)))']*BETA_contextsgbow;
    contextordersgbow_pred1 = [1 squeeze(zscore(lo_response2_450(i,1,:)))']*BETA_contextordersgbow;
    contextordersgbow_pred2 = [1 squeeze(zscore(lo_response2_450(i,2,:)))']*BETA_contextordersgbow;
    
    %get the values from data

    actual_context_vector1 = squeeze(zscore(lo_context_svd(i,1,:)));
    actual_context_vector2 = squeeze(zscore(lo_context_svd(i,2,:)));
    actual_order_vector1 = squeeze(zscore(lo_order_svd(i,1,:)));
    actual_order_vector2 = squeeze(zscore(lo_order_svd(i,2,:)));
    actual_contextorder_vector1 = squeeze(zscore(lo_contextorder_svd(i,1,:)));
    actual_contextorder_vector2 = squeeze(zscore(lo_contextorder_svd(i,2,:)));
    actual_sg_vector1 = squeeze(zscore(lo_sg_svd(i,1,:)));
    actual_sg_vector2 = squeeze(zscore(lo_sg_svd(i,2,:)));
    actual_bow_vector1 = squeeze(zscore(lo_bow_svd(i,1,:)));
    actual_bow_vector2 = squeeze(zscore(lo_bow_svd(i,2,:)));
    actual_sgbow_vector1 = squeeze(zscore(lo_sg_svd(i,1,:)));
    actual_sgbow_vector2 = squeeze(zscore(lo_sg_svd(i,2,:)));
    actual_contextsgbow_vector1 = [squeeze(zscore(lo_context_svd(i,1,:)))' squeeze(zscore(lo_sg_svd(i,1,:)))' squeeze(zscore(lo_bow_svd(i,1,:)))'];
    actual_contextsgbow_vector2 = [squeeze(zscore(lo_context_svd(i,2,:)))' squeeze(zscore(lo_sg_svd(i,2,:)))' squeeze(zscore(lo_bow_svd(i,2,:)))'];
    actual_contextordersgbow_vector1 = [squeeze(zscore(lo_context_svd(i,1,:)))' squeeze(zscore(lo_order_svd(i,1,:)))' squeeze(zscore(lo_sg_svd(i,1,:)))' squeeze(zscore(lo_bow_svd(i,1,:)))'];
    actual_contextordersgbow_vector2 = [squeeze(zscore(lo_context_svd(i,2,:)))' squeeze(zscore(lo_order_svd(i,2,:)))' squeeze(zscore(lo_sg_svd(i,2,:)))' squeeze(zscore(lo_bow_svd(i,2,:)))'];
    %test using cosine similarities

    cor_context = dot(context_pred1, actual_context_vector1)/(norm(context_pred1)*norm(actual_context_vector1)) + dot(context_pred2, actual_context_vector2)/(norm(context_pred2)*norm(actual_context_vector2));
    inc_context = dot(context_pred1, actual_context_vector2)/(norm(context_pred1)*norm(actual_context_vector2)) + dot(context_pred2, actual_context_vector1)/(norm(context_pred2)*norm(actual_context_vector1));
    if(cor_context > inc_context)
        correct.('context') = correct.('context') + 1;
    end
    cor_order = dot(order_pred1, actual_order_vector1)/(norm(order_pred1)*norm(actual_order_vector1)) + dot(order_pred2, actual_order_vector2)/(norm(order_pred2)*norm(actual_order_vector2));
    inc_order = dot(order_pred1, actual_order_vector2)/(norm(order_pred1)*norm(actual_order_vector2)) + dot(order_pred2, actual_order_vector1)/(norm(order_pred2)*norm(actual_order_vector1));
    if(cor_order > inc_order)
        correct.('order') = correct.('order') + 1;
    end
    cor_contextorder = dot(contextorder_pred1, actual_contextorder_vector1)/(norm(contextorder_pred1)*norm(actual_contextorder_vector1)) + dot(contextorder_pred2, actual_contextorder_vector2)/(norm(contextorder_pred2)*norm(actual_contextorder_vector2));
    inc_contextorder = dot(contextorder_pred1, actual_contextorder_vector2)/(norm(contextorder_pred1)*norm(actual_contextorder_vector2)) + dot(contextorder_pred2, actual_contextorder_vector1)/(norm(contextorder_pred2)*norm(actual_contextorder_vector1));
    if(cor_contextorder > inc_contextorder)
        correct.('contextorder') = correct.('contextorder') + 1;
    end
    cor_sg = dot(sg_pred1, actual_sg_vector1)/(norm(sg_pred1)*norm(actual_sg_vector1)) + dot(sg_pred2, actual_sg_vector2)/(norm(sg_pred2)*norm(actual_sg_vector2));
    inc_sg = dot(sg_pred1, actual_sg_vector2)/(norm(sg_pred1)*norm(actual_sg_vector2)) + dot(sg_pred2, actual_sg_vector1)/(norm(sg_pred2)*norm(actual_sg_vector1));
    if(cor_sg > inc_sg)
        correct.('sg') = correct.('sg') + 1;
    end
    cor_bow = dot(bow_pred1, actual_bow_vector1)/(norm(bow_pred1)*norm(actual_bow_vector1)) + dot(bow_pred2, actual_bow_vector2)/(norm(bow_pred2)*norm(actual_bow_vector2));
    inc_bow = dot(bow_pred1, actual_bow_vector2)/(norm(bow_pred1)*norm(actual_bow_vector2)) + dot(bow_pred2, actual_bow_vector1)/(norm(bow_pred2)*norm(actual_bow_vector1));
    if(cor_bow > inc_bow)
        correct.('bow') = correct.('bow') + 1;
    end
    cor_sgbow = dot(sgbow_pred1, actual_sgbow_vector1)/(norm(sgbow_pred1)*norm(actual_sgbow_vector1)) + dot(sgbow_pred2, actual_sgbow_vector2)/(norm(sgbow_pred2)*norm(actual_sgbow_vector2));
    inc_sgbow = dot(sgbow_pred1, actual_sgbow_vector2)/(norm(sgbow_pred1)*norm(actual_sgbow_vector2)) + dot(sgbow_pred2, actual_sgbow_vector1)/(norm(sgbow_pred2)*norm(actual_sgbow_vector1));
    if(cor_sgbow > inc_sgbow)
        correct.('sgbow') = correct.('sgbow') + 1;
    end
    
    cor_contextsgbow = dot(contextsgbow_pred1, actual_contextsgbow_vector1)/(norm(contextsgbow_pred1)*norm(actual_contextsgbow_vector1)) + dot(contextsgbow_pred2, actual_contextsgbow_vector2)/(norm(contextsgbow_pred2)*norm(actual_contextsgbow_vector2));
    inc_contextsgbow = dot(contextsgbow_pred1, actual_contextsgbow_vector2)/(norm(contextsgbow_pred1)*norm(actual_contextsgbow_vector2)) + dot(contextsgbow_pred2, actual_contextsgbow_vector1)/(norm(contextsgbow_pred2)*norm(actual_contextsgbow_vector1));
    if(cor_contextsgbow > inc_contextsgbow)
        correct.('contextsgbow') = correct.('contextsgbow') + 1;
    end
    
    cor_contextordersgbow = dot(contextordersgbow_pred1, actual_contextordersgbow_vector1)/(norm(contextordersgbow_pred1)*norm(actual_contextordersgbow_vector1)) + dot(contextordersgbow_pred2, actual_contextordersgbow_vector2)/(norm(contextordersgbow_pred2)*norm(actual_contextordersgbow_vector2));
    inc_contextordersgbow = dot(contextordersgbow_pred1, actual_contextordersgbow_vector2)/(norm(contextordersgbow_pred1)*norm(actual_contextordersgbow_vector2)) + dot(contextordersgbow_pred2, actual_contextordersgbow_vector1)/(norm(contextordersgbow_pred2)*norm(actual_contextordersgbow_vector1));
    if(cor_contextordersgbow > inc_contextordersgbow)
        correct.('contextordersgbow') = correct.('contextordersgbow') + 1;
    end
end

save_path = strcat('/gpfs/group/pul8/default/read/',par,'/fMRI_Analyses/Conceptual_Change/results_svd_sv2_450_vox2vec.mat');
save(save_path,'correct','N');


