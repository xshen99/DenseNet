
local t = require 'datasets/transforms'

local M = {}
local LungROIDataset = torch.class('resnet.LungROIDataset', M)

function LungROIDataset:__init(imageInfo, opt, split)
   assert(imageInfo[split], split)
   self.imageInfo = imageInfo[split]
   self.split = split
end

function LungROIDataset:get(i)
   local image = self.imageInfo.data[i]:float()
   local label = self.imageInfo.labels[i]

   return {
      input = image,
      target = label,
   }
end

function LungROIDataset:size()
   return self.imageInfo.data:size(1)
end

-- Computed from entire training set
local meanstd = {
   mean = {169.5, 190.1, 62.1},
   std  = {82.7,  81.7,  109.5},
}

function LungROIDataset:preprocess()
   if self.split == 'train' then
      return t.Compose{
         t.ColorNormalize(meanstd),
         t.HorizontalFlip(0.5),
         t.RandomCrop(32, 4),
      }
   elseif self.split == 'val' then
      return t.ColorNormalize(meanstd)
   else
      error('invalid split: ' .. self.split)
   end
end

return M.LungROIDataset
