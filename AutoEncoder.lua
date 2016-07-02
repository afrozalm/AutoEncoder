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

encoder1 = nn.Sequential()
encoder2 = nn.Sequential()
encoder = nn.Sequential()

pool1 = nn.SpatialMaxPooling( 2, 2, 2, 2 )
pool2 = nn.SpatialMaxPooling( 2, 2, 2, 2 )

encoder1:add( nn.SpatialConvolution( 1, 6, 5, 5 ) )  -- 1x32x32  -> 6x28x28
encoder1:add( pool1 )                                -- 6x28x28  -> 6x14x14
encoder1:add( nn.ReLU() )
encoder1:add( nn.SpatialConvolution( 6, 16, 5, 5 ) ) -- 6x14x14  -> 16x10x10
encoder1:add( pool2 )                                -- 16x10x10 -> 16x5x5
encoder1:add( nn.ReLU() )
encoder1:add( nn.View( 16*5*5 ) )
encoder1:add( nn.Linear( 16*5*5, 120 ) )
encoder1:add( nn.Tanh() )                            -- Latent variables

encoder2:add( nn.Linear( 120, 84 ) )
encoder2:add( nn.Tanh() )
encoder2:add( nn.Linear( 84, 10 ) )
encoder2:add( nn.LogSoftMax() )

encoder:add( encoder1 )
encoder:add( encoder2 )

criterion = nn.ClassNLLCriterion()                  -- given ( x, label ), NLL minimises -x[label]

params, gradParams = encoder:getParameters()

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
    
    -- setting inputs and targets for minibatches
    for minibatch_number = 1, trainset.size, batchsize do
        local size = math.min( trainset.size - minibatch_number + 1, batchsize )
        local inputs = torch.Tensor(size, 1, 32, 32)
        local targets = torch.Tensor(size)
        
        for index = 1, size do
            inputs[index] = trainset.data[ shuffle[ index + minibatch_number - 1 ]]
            targets[index] = trainset.labels[ shuffle[ index + minibatch_number - 1 ]]
        end

        -- defining feval function to return loss and gradients of loss w.r.t. params
        feval = function( new_params )
            collectgarbage()
            if params ~= new_params then params:copy(new_params) end
            
            -- initializing gradParsams to zero
            gradParams:zero()

            -- calculating loss and param gradients
            local outputs = encoder:forward( inputs )
            local loss = criterion:forward( outputs, targets )
            encoder:backward( inputs, criterion:backward( outputs, targets ) )

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
        current_loss = current_loss + criterion:forward( encoder:forward( validationset.data[i] ), validationset.labels[i]  )
        count = count + 1
    end
    return current_loss / count
end

-- Training
max_epoches = 75
increasing = 0
prev_loss = 0
validation_loss = 0
for i = 1, max_epoches do
    print('Epoch : ' .. i .. '\n')
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
    print('Encoder Validation Loss : ' .. validation_loss .. '\n')
    print('_____________________________________________________' .. '\n')
end

filename = 'ConvEnc-' .. max_epoches .. '.net'
torch.save( filename, encoder )

decoder = nn.Sequential()

decoder:add( encoder1 )
decoder:add( nn.Linear( 120, 16*5*5 ) )                 -- 120 -> 16*5*5
decoder:add( nn.Tanh() )
decoder:add( nn.View( 16, 5, 5 ) )                      -- 16*5*5 -> 16x5x5
decoder:add( nn.SpatialMaxUnpooling( pool2 ) )          -- 16x5x5 -> 16x10x10
decoder:add( nn.Tanh() )
decoder:add( nn.SpatialFullConvolution( 16, 6, 5, 5 ) ) -- 16x10x10 -> 6x14x14
decoder:add( nn.SpatialMaxUnpooling( pool1 ) )          -- 6x14x14 -> 6x28x28
decoder:add( nn.Tanh() )
decoder:add( nn.SpatialFullConvolution( 6, 1, 5, 5 ) )  -- 6x28x28 -> 1x32x32
dec_criterion = nn.MSECriterion()

params, gradParams = decoder:getParameters()

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
max_epoches = 75
increasing = 0
prev_loss = 0
validation_loss = 0
for i = 1, max_epoches do
    print('Epoch : ' .. i .. '\n')
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
    print('Decoder Validation Loss : ' .. validation_loss .. '\n')
    print('_____________________________________________________' .. '\n')
end

filename = 'ConvDec-' .. max_epoches .. '.net'
torch.save( filename, decoder )
