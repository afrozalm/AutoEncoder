require 'image';
require 'nn';
require 'torch';

testPath = '/home1/mnist.t7/test_32x32.t7'
testset = torch.load(testPath,'ascii')

cmd = torch.CmdLine()
cmd:option('-index', 1, 'initial index')
cmd:option('-ones',0, 'all ones matrix')
cmd:option('-value',0, 'all ones matrix')
cmd:option('-model','ConvDec-75.net','model name')
params = cmd:parse(arg)

index = params['index']

model = torch.load('/home1/afrozalm/' .. params['model'] )

if params['ones'] == 1 then
    input = torch.Tensor(1,32,32):fill(params['value']):double()
else
    input = testset.data[index]:double()
end

output =  model:forward(input)
for i = 1,32 do
    for j = 1, 32 do
        if output[1][i][j] < 70 then output[1][i][j] = 0 end
        if output[1][i][j] > 155 then output[1][i][j] = 255 end
    end
end
insavename = tostring(index) .. '_input_index_.jpg'
ousavename = tostring(index) .. '_output_index_.jpg'
image.save( insavename,input)
image.save( ousavename,output)
