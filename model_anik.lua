--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'nn'
require 'cunn'
require 'optim'
require 'loadcaffe'

--[[
   1. Create Model
   2. Create Criterion
   3. If preloading option is set, preload weights from existing models appropriately
   4. Convert model to CUDA
]]--

-- 1.1. Create Network
local config = opt.netType .. '_' .. opt.backend
paths.dofile('models/' .. config .. '.lua')
print('=> Creating model from file: models/' .. config .. '.lua')
model = createModel(opt.nGPU) -- for the model creation code, check the models/ folder

-- 2. Create Criterion
criterion = nn.ClassNLLCriterion()

print('=> Model')
print(model)

print('=> Criterion')
print(criterion)

-- 3. If preloading option is set, preload weights from existing models appropriately
if opt.retrain ~= 'none' then
   assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
   print('Loading model from file: ' .. opt.retrain);
   model = torch.load(opt.retrain)
end

-- 4. Convert model to CUDA
print('==> Converting model to CUDA')
-- model = model:cuda()
prototxt_name = '/home/anik/external_models/bvlc_alexnet_deploy.prototxt'
binary_name = '/home/anik/external_models/bvlc_alexnet.caffemodel'
model = loadcaffe.load(prototxt_name, binary_name, 'cudnn')
model:remove(24);
model:remove(23);
model:add(nn.Linear(4096,2))
model:add(nn.LogSoftMax())
model = model:cuda()
criterion:cuda()
-- debugger.enter()
collectgarbage()
