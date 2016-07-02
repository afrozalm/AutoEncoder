require 'image';
require 'nn';
require 'torch';

testPath = '/home1/mnist.t7/train_32x32.t7'
testset = torch.load(testPath,'ascii')

cmd = torch.CmdLine()
cmd:option('-model','ConvDec-75.net','model name')
params = cmd:parse(arg)

model = torch.load('/home1/afrozalm/' .. params['model'] )
max = 0
maxSalient = 0
SalientInput = torch.DoubleTensor(1, 32, 32)
SalientOutput = torch.DoubleTensor(1, 32, 32)
criterion = nn.MSECriterion()
for i = 1, 50000 do
    input = testset.data[i]:double()
    output =  model:forward(input)
    for i = 1,32 do
        for j = 1, 32 do
            if output[1][i][j] < 70 then output[1][i][j] = 0 end
            if output[1][i][j] > 155 then output[1][i][j] = 255 end
        end
    end
    err = criterion:forward( output, input )
    if err > max then
        max = err
        maxSalient = i
        SalientInput = input
        SalientOutput = output
    end
end
insavename = tostring(maxSalient) .. '_inputSalient.jpg'
ousavename = tostring(maxSalient) .. '_outputSalient.jpg'
image.save( insavename,SalientInput)
image.save( ousavename,SalientOutput)
