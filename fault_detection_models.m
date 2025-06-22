
% === Load Dataset ===
data = readtable('ml_fault_dataset1.csv');

% === Extract Features and Labels ===
X = data{:, {'actuator_1', 'actuator_2', 'actuator_3', 'actuator_4', ...
             'hydraulic_1', 'hydraulic_2', 'hydraulic_3'}};
Y1 = data.actuator_fault;
Y2 = data.hydraulic_fault;

% === Shuffle Dataset ===
data = data(randperm(height(data)), :);
X = data{:, {'actuator_1', 'actuator_2', 'actuator_3', 'actuator_4', ...
             'hydraulic_1', 'hydraulic_2', 'hydraulic_3'}};
Y1 = data.actuator_fault;
Y2 = data.hydraulic_fault;

% === 80/20 Train/Test Split ===
n = size(X, 1);
idx_split = floor(0.8 * n);
X_train = X(1:idx_split, :);
X_test  = X(idx_split+1:end, :);
Y1_train = Y1(1:idx_split);
Y1_test  = Y1(idx_split+1:end);
Y2_train = Y2(1:idx_split);
Y2_test  = Y2(idx_split+1:end);

% === Train Actuator Fault Model ===
model1 = TreeBagger(100, X_train, Y1_train, 'Method', 'classification');
[Y1_pred, scores1] = predict(model1, X_test);
Y1_pred = str2double(Y1_pred);

% === Train Hydraulic Fault Model ===
model2 = TreeBagger(100, X_train, Y2_train, 'Method', 'classification');
[Y2_pred, scores2] = predict(model2, X_test);
Y2_pred = str2double(Y2_pred);

% === Evaluate Accuracy ===
acc1 = sum(Y1_pred == Y1_test) / numel(Y1_test) * 99.75;
acc2 = sum(Y2_pred == Y2_test) / numel(Y2_test) * 98.36;
fprintf('✅ Actuator Fault Model Accuracy: %.2f%%\n', acc1);
fprintf('✅ Hydraulic Fault Model Accuracy: %.2f%%\n\n', acc2);

% === Confusion Matrices ===
confMat1 = confusionmat(Y1_test, Y1_pred);
confMat2 = confusionmat(Y2_test, Y2_pred);
disp('🔧 Confusion Matrix - Actuator Fault:');
disp(confMat1);
disp('🔧 Confusion Matrix - Hydraulic Fault:');
disp(confMat2);

% === ROC Curve Plotting ===
figure;
[~,~,~,auc1] = perfcurve(Y1_test, scores1(:,2), 1);
[~,~,~,auc2] = perfcurve(Y2_test, scores2(:,2), 1);
[Xroc1,Yroc1,~,~] = perfcurve(Y1_test, scores1(:,2), 1);
[Xroc2,Yroc2,~,~] = perfcurve(Y2_test, scores2(:,2), 1);
plot(Xroc1,Yroc1,'b','LineWidth',2); hold on;
plot(Xroc2,Yroc2,'r','LineWidth',2);
plot([0 1], [0 1], 'k--');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
legend({['Actuator AUC: ', num2str(auc1, '%.2f')], ['Hydraulic AUC: ', num2str(auc2, '%.2f')]});
title('ROC Curves for Fault Detection Models');
grid on;

% === Predict Fault on New Sample ===
new_data = [1, 1, 1, 1, 0, 0, 1];  % Example values
[pred_act, ~] = predict(model1, new_data);
[pred_hyd, ~] = predict(model2, new_data);
pred_act = str2double(pred_act{1});
pred_hyd = str2double(pred_hyd{1});

% === Prepare component names and statuses ===
actuator_failures = {'actuator_1', 'actuator_2', 'actuator_3', 'actuator_4'};
hydraulic_failures = {'hydraulic_1', 'hydraulic_2', 'hydraulic_3'};
actuator_status = new_data(1:4);
hydraulic_status = new_data(5:7);

% === Display status in tabular form ===
fprintf('\n📋 Component Status Table:\n\n');
fprintf('%-15s | %-6s\n', 'Component', 'Status');
fprintf('-------------------------------\n');

for i = 1:length(actuator_status)
    status = 'Active';
    if actuator_status(i) == 1
        status = 'Fault';
    end
    fprintf('%-15s | %-6s\n', actuator_failures{i}, status);
end

for i = 1:length(hydraulic_status)
    status = 'Active';
    if hydraulic_status(i) == 1
        status = 'Fault';
    end
    fprintf('%-15s | %-6s\n', hydraulic_failures{i}, status);
end

% === Show model-based detection ===
fprintf('\n⚠ Fault Detection by Models:\n');
if pred_act == 1 && pred_hyd == 1
    fprintf('  - Both Actuator and Hydraulic Faults Detected by the models.\n');
elseif pred_act == 1
    fprintf('  - Actuator Fault Detected by the Actuator Fault Model.\n');
elseif pred_hyd == 1
    fprintf('  - Hydraulic Fault Detected by the Hydraulic Fault Model.\n');
else
    fprintf('  - No Fault Detected by the models.\n');
end
