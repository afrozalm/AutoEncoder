require 'image';
require 'nn';
require 'torch';
require 'optim';

trainPath = '/home1/mnist.t7/train_32x32.t7'
fullset = torch.load(trainPath,'ascii')

dataset = torch.Tensor( 6000, 1, 32, 32 )
for i = 1,6000 do
    dataset[i] = fullset.data[i]:double()
end

trainset = {
    size = 5000,
    data = dataset[{{1,5000}}],
    labels = fullset.labels[{{1, 5000}}]
}

validationset = {
    size = 1000,
    data = dataset[{{5001, 6000}}],
    labels = fullset.labels[{{5001, 6000}}]
}


modelname = '/home1/afrozalm/ConvDec-75.net'
decoder = torch.load( modelname )
params, gradParams = decoder:getParameters()

dec_criterion = nn.MSECriterion()
optimState = {
   learningRate = 1e-2,
   learningRateDecay = 1e-4,
   weightDecay = 1e-3,
   momentum = 1e-4
}

-- step runs sgd over trainset and returns average loss over minibatche
step = function ( batchsize )
    
    -- setting up variables
    local count = 0
    local current_loss = 0
    local shuffle = torch.randperm(trainset.size)
    
    -- setting default batchsize as 70
    batchsize = batchsize or 70
    
    -- setting inputs for minibatches
    for minibatch_number = 1, trainset.size, batchsize do
        local size = math.min( trainset.size - minibatch_number + 1, batchsize )
        local inputs = torch.Tensor(size, 1, 32, 32)
 
        for index = 1, size do
            inputs[index] = trainset.data[ shuffle[ index + minibatch_number - 1 ]]
        end

        -- defining feval function to return loss and gradients of loss w.r.t. params
        feval = function( new_params )
            collectgarbage()
            if params ~= new_params then params:copy(new_params) end
            
            -- initializing gradParsams to zero
            gradParams:zero()

            -- calculating loss and param gradients
            local outputs = decoder:forward( inputs )
            local loss = dec_criterion:forward( outputs, inputs )
            decoder:backward( inputs, dec_criterion:backward( outputs, inputs ) )

            return loss, gradParams
        end
        
        -- getting loss
        -- optim returns x*, {fx} where x* is new set of params and {fx} is { loss } => fs[ 0 ]  carries loss from feval
        _, fs = optim.sgd(feval, params, optimState)
        current_loss = fs[ 1 ] + current_loss
        count = count + 1
    end    
    return current_loss / count
end

-- Evaluating loss over validation set
eval = function( )
    local count = 0
    local current_loss = 0
    local size = validationset.size
    for i = 1, size do
        current_loss = current_loss + dec_criterion:forward( decoder:forward( validationset.data[i] ), validationset.data[i]  )
        count = count + 1
    end
    return current_loss / count
end

-- Training
max_epoches = 95
increasing = 0
prev_loss = 0
validation_loss = 0
print( 'further training ' .. modelname )
for i = 1, max_epoches do
    print('Epoch : ' .. i .. '\r')
    if i < 70 then step( i + 1 ) else step ( 70 ) end
    validation_loss = eval()
    if prev_loss < validation_loss and increasing == 5 then 
        break
    elseif prev_loss < validation_loss then
        prev_loss = validation_loss
        increasing = increasing + 1
    else 
        prev_loss = validation_loss
        increasing = 0
    end
    print('Decoder Validation Loss : ' .. validation_loss .. '\r')
    print('_____________________________________________________' .. '\r')
end

torch.save( modelname, decoder )
