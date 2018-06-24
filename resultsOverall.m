%% Results

addpath('./libs');
path = './assets/results/model1.13-Jun-2018.18-55-53';
viz = Viz(path);
%%
viz.plotNet();
viz.plotData(1:50);
viz.plotCosts();
viz.plotParameters();
viz.plotExpectedActualOutputs();
% viz.playFilterVideo();
%%
% epoch number
epoch = 1;

% training data
train = viz.dataIndexes.train;
x = viz.X(train);
y = viz.Y(train);

% parameters (filters, biases)
w = viz.params.('w_B').history{epoch};
w = 0.1 * w;
w_ = w(end:-1:1);
b = viz.params.('b_B').history{epoch};

% mapping
f = @(x) max(0, conv(x, w_, 'valid') + b);
% distance
d = @(a, b) norm(a - b, 1);
% error
DataUtils.error(x, y, f, d)

% viz.costs.train(epoch)
viz.dag.load_epoch(epoch);
viz.dag.get_train_cost()

y = viz.dag.out(x);
net = viz.dag.net;
w = net.getParam('w_B').value;
b = net.getParam('b_B').value;
w_ = w(end:-1:1);
f = @(x) max(0, conv(x, w_, 'valid') + b);
DataUtils.error(x, y, f, d)