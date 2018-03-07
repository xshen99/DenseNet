
local URL = 'https://storage.googleapis.com/densenet/lungDataset.tar.gz'

local M = {}

local function convertToTensor(files)
   local data, labels

   for _, file in ipairs(files) do
      local m = torch.load(file)
      if not data then
         data = m.data:t()
         labels = m.labels:squeeze()
      else
         data = torch.cat(data, m.data:t(), 1)
         labels = torch.cat(labels, m.labels:squeeze())
      end
   end

   return {
      data = data:contiguous():view(-1, 3, 32, 32),
      labels = labels,
   }
end

function M.exec(opt, cacheFile)
   print("=> Downloading lungROI dataset from " .. URL)
   local ok = os.execute('curl ' .. URL .. ' | tar xz -C gen/')
   assert(ok == true or ok == 0, 'error downloading lungROI dataset')

   print(" | combining dataset into a single file")
   local trainData = convertToTensor({
      'gen/lungDataset.tar/training.t7',
   local testData = convertToTensor({
      'gen/lungDataset/testing.t7',
   })

   print(" | saving lungROI dataset to " .. cacheFile)
   torch.save(cacheFile, {
      train = trainData,
      val = testData,
   })
end

return M
