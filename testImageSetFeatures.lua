1;95;0crequire 'cudnn'
require 'cunn'
require 'nn'
local gm = assert(require 'graphicsmagick')
require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'image'
require 'inn'

-- PARAMETERS
-- Retrain
-- modelFile = "/home/anik/fbcunn_imagenet/imagenet_runs/alexnet12/,MonOct1221:59:202015/model_1.t7"
-- New network
modelFile = "/home/i-akashg/fbcunn_imagenet/imagenet_runs/alexnet12/,ThuDec318:32:482015/model_15.t7"
--modelFile = 
-- Test
--modelFile = '/usr/local/torch/fbcunn/examples/imagenet/models/alexnet_fbcunn.lua'
dirName = arg[5] ---"/home/i-akashg/one_img" -- images you want features of
outputFilePath = arg[6] ---"/home/i-akashg/imageFeatures"
useJitter = 0

print("Model: ", modelFile)
print("Input Dir: ", dirName)
print("Output file: ", outputFilePath)

local tm = torch.Timer()
local opts = paths.dofile('opts.lua')
a = {}
for i=0,4 do
	a[i] = arg[i]
end
opt = opts.parse(a)

-- debugger = require('fb.debugger')
-- debugger.enter()

-- local trainCache = paths.concat(opt.cache, 'trainCache.t7')
-- local testCache = paths.concat(opt.cache, 'testCache.t7')
--local meanstdCache = paths.concat(opt.cache, 'meanstdCache.t7')

local loadSize   = {3, 256, 256}
local sampleSize = {3, 224, 224}
local mean,std
--local meanstd = torch.load(meanstdCache)
mean = 0
std = 0
print('Loaded mean and std from cache.')

local oH = sampleSize[2]
local oW = sampleSize[3];
local numCrops
if useJitter == 1 then
	numCrops = 10
else
	numCrops = 4
end
local out = torch.Tensor(numCrops, 3, oW, oH)

--paths.dofile(modelFile)
--model = createModel(opt.nGPU)

model =torch.load(modelFile)
model:remove(24)
model:remove(23)
pf = io.popen('find "'..dirName..'" -type f')

-- io.output(resultsFile)
fileCounter = 1
for path in pf:lines() do
    	local file_name
    	for i in string.gmatch(path, "([^/]+)") do
  	    file_name = i
	end
    	local resultsFile = io.open(outputFilePath..'/'..file_name, "w")
	if (fileCounter % 100) == 0 then print("Processing file ", fileCounter) end

        -- print(path)
       -- resultsFile:write(path)
        local input = gm.Image():load(path, loadSize[3], loadSize[2])

        -- find the smaller dimension, and resize it to 256 (while keeping aspect ratio)
        local iW, iH = input:size()
        if iW < iH then
           input:size(256, 256 * iH / iW);
        else
           input:size(256 * iW / iH, 256);
        end
        iW, iH = input:size();
        local im = input:toTensor('float','RGB','DHW')
	local rotated_im = im:clone()
	rotated_im = rotated_im:transpose(2,3) -- ASSUMES A SQUARE
		
        -- mean/std
--        for i=1,3 do -- channels
--           if mean then im[{{i},{},{}}]:add(-mean[i]) end
--	   if mean then rotated_im[{{i},{},{}}]:add(-mean[i]) end
--           if  std then im[{{i},{},{}}]:div(std[i]) end
--	   if  std then rotated_im[{{i},{},{}}]:div(std[i]) end
--        end

        local w1 = math.ceil((iW-oW)/2)
        local h1 = math.ceil((iH-oH)/2)
        out[1] = image.crop(im, w1, h1, w1+oW, h1+oW) -- center patch
	out[2] = image.hflip(out[1])	
--	out[3] = image.crop(rotated_im, w1, h1, w1+oW, h1+oW)
--	out[4] = image.hflip(out[3])
	
        if useJitter == 1 then
			h1 = 1; w1 = 1;
			out[3] = image.crop(im, w1, h1, w1+oW, h1+oW)  -- top-left
			out[4] = image.hflip(out[3])
			h1 = 1; w1 = iW-oW;
			out[5] = image.crop(im, w1, h1, w1+oW, h1+oW)  -- top-right
			out[6] = image.hflip(out[5])
			h1 = iH-oH; w1 = 1;
			out[7] = image.crop(im, w1, h1, w1+oW, h1+oW)  -- bottom-left
			out[8] = image.hflip(out[7])
			h1 = iH-oH; w1 = iW-oW;
			out[9] = image.crop(im, w1, h1, w1+oW, h1+oW)  -- bottom-right
			out[10] = image.hflip(out[9])
		end

        img = out
        model:evaluate()
        predictions = model:forward(img:cuda())
	-- model:remove(24)
	--print(model)
        for py=1,1 do
           for px=1,4096 do
	      -- print(predictions)
              resultsFile:write("\n",predictions[py][px])
           end
        end
        resultsFile:write("\n")
	fileCounter = fileCounter + 1
end
--io.close(resultsFile)
print('Predictions done. Time = %.2f', tm:time().real)
