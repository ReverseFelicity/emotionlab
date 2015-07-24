% test cross validation
path_smooth = 'E:\wsy\Lab\data\feature_smooth';
% path_result = 'G:\zwl\Emotion\data\chinese\result';
path_result = 'E:\wsy\Lab\data\result_cv';
fileList={...
    'dujingcheng_20131027','dujingcheng_20131030','dujingcheng_20131107',...
    'mahaiwei_20130712','mahaiwei_20131016','mahaiwei_20131113',...
    'penghuiling_20131027','penghuiling_20131030','penghuiling_20131106',...
    'zhujiayi_20130709','zhujiayi_20131016','zhujiayi_20131105',...
    'wuyangwei_20131127','wuyangwei_20131201','wuyangwei_20131207',...
    'weiwei_20131130','weiwei_20131204','weiwei_20131211',...
    'jianglin_20140404','jianglin_20140413','jianglin_20140419',...
    'liuye_20140411','liuye_20140418','liuye_20140506',...
    'sunxiangyu_20140511','sunxiangyu_20140514','sunxiangyu_20140521',...
    'xiayulu_20140527','xiayulu_20140603','xiayulu_20140610',...
    'jingjing_20140603','jingjing_20140611','jingjing_20140629',...
    'yansheng_20140601','yansheng_20140615','yansheng_20140627',...
    'wusifan_20140618','wusifan_20140625','wusifan_20140630',...
    'wangkui_20140620','wangkui_20140627','wangkui_20140704',...
    'liuqiujun_20140621','liuqiujun_20140702','liuqiujun_20140705',...
    };
label = [1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1];
result_file_name = sprintf('%s\\%s.txt',path_result,'meanscaleXLDS_SVM');
fid_result = fopen(result_file_name,'wt');
fileNum = length(fileList);
feature_name_list = {'psd','de','dasm','rasm','asm','dcau'};
dim_list = [62,62,27,27,54,23]; %dimension
fprintf(fid_result,'Feature\tTotal\tDelta\tTheta\tAlpha\tBeta\tGamma\n');
for i=1:fileNum
    filename = fileList{i};
    fea_smooth_file = sprintf('%s\\%s.mat', path_smooth,filename);
    fprintf(fid_result,'%s\n',filename);
    disp(sprintf('processing %s ...\n', fea_smooth_file));
    load (fea_smooth_file);
    for p = 1:length(feature_name_list)
        feature_name = feature_name_list{p};
        fprintf(fid_result,'%s',feature_name);
        dim = dim_list(p);
        %5 frequency bands
        for j = 0:5  %%%%%%%%%%%%%%%%%%%%
            %prepare train data
            if j ~= 0
                train_inst = zeros(dim,0);
            else
                train_inst = zeros(dim*5, 0);
            end
            train_label = zeros(1,0);
            %9 cases    train*****************************
            
            aaa = randperm(15);
            for iii = 1:5
                eval(['ccc(',num2str(iii),',:)=aaa(',num2str(1+3*(iii-1)),':',num2str(3+3*(iii-1)),')']);
            end
            resultsum = 0;
            for i_case = 1:5
                testbox = ccc(i_case,:);
                trainbox = zeros(1,0);
                for j_case = 1:5
                    if j_case ~=i_case
                        trainbox = [trainbox ccc(j_case,:)]; 
                    end
                end
                disp(trainbox);
                disp(testbox);
                for tt = 1:length(trainbox)
                    t =trainbox(tt); 
                    eval(['feature = ', feature_name, '_LDS',num2str(t), ';']);
                    [use_less case_num] = size(feature(:,:,1));
                    if j ~= 0
                        train_inst = [train_inst feature(:,:,j)];
                    else
                        temp_inst = [];
                        for fre = 1:5
                            temp_inst = [temp_inst;feature(:,:,fre)];
                        end
                        train_inst = [train_inst temp_inst];
                    end
                    train_label = [train_label label(t)*ones(1,case_num)];
                end
                %prepare test data
                if j ~= 0 
                    test_inst = zeros(dim, 0);
                else
                    test_inst = zeros(dim*5, 0);
                end
                test_label = zeros(1, 0);
                %5 cases   test**********************************
                for tt = 1:length(testbox)
                    t =testbox(tt); 
                    eval(['feature = ', feature_name, '_LDS',num2str(t), ';']);
                    [use_less case_num] = size(feature(:,:,1));
                    if j ~= 0
                        test_inst = [test_inst feature(:,:,j)];
                    else
                        temp_inst = [];
                        for fre = 1:5
                            temp_inst = [temp_inst;feature(:,:,fre)];
                        end
                        test_inst = [test_inst temp_inst];
                    end
                    test_label = [test_label label(t)*ones(1,case_num)];
                end
    %             %scalenorm
                [tt vv] = size(train_inst);
                for vvv = 1:vv
                    min_num = min(train_inst(:,vvv));
                    max_num = max(train_inst(:,vvv));
                    train_inst(:,vvv) = 10*(train_inst(:,vvv) - min_num)/(max_num-min_num);
                end
                [tt vv] = size(test_inst);
                for vvv = 1:vv
                    min_num = min(test_inst(:,vvv));
                    max_num = max(test_inst(:,vvv));
                    test_inst(:,vvv) = 10*(test_inst(:,vvv) - min_num)/(max_num-min_num);
                end
                %meannorm
                [tt vv] = size(train_inst);
                for vvv = 1:vv
                    ave_num = mean(train_inst(:,vvv));
                    if j ~= 0
                        train_inst(:,vvv) = train_inst(:,vvv) - ave_num*ones(dim,1);
                    else
                        train_inst(:,vvv) = train_inst(:,vvv) - ave_num*ones(dim*5,1);
                    end
                end
                [tt vv] = size(test_inst);
                for vvv = 1:vv
                    ave_num = mean(test_inst(:,vvv));
                    if j ~= 0
                        test_inst(:,vvv) = test_inst(:,vvv) - ave_num*ones(dim,1);
                    else
                        test_inst(:,vvv) = test_inst(:,vvv) - ave_num*ones(dim*5,1);
                    end
                end
                % svm test
                temp_result = 0;
                % choose best parameter.
                for c = -10:10
                    para=sprintf('-s 3 -c %f',2^c);   % -s 2 may be faster
                    model = train(train_label', sparse(train_inst'), para);
                    %model = svmtrain(train_label', train_inst', '-b 1');
                    [a p b] = predict(test_label', sparse(test_inst'), model);
                    if p(1) > temp_result
                        temp_result = p(1);
                    end
                end
                resultsum = resultsum + temp_result;
            end
            fprintf(fid_result,'\t%.2f',resultsum/5);
        end
        fprintf(fid_result,'\n');
    end
end
fclose(fid_result);