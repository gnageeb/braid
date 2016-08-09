require 'data'
require 'nn'
require 'cunn'
require 'rnn'
cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training a simple network')
cmd:text()
cmd:text('Options')
cmd:option('-L1','english','First Language')
cmd:option('-L2','turkish','Second Language')
cmd:option('-datadir','./data/train_en_tr/','Training data directory')
cmd:option('-lr',0.1,'Learning Rate')
cmd:option('-lr_decay',1,'Learning rate decay')
cmd:option('-threshold',10,'Decay threshold')
cmd:option('-embedding_size',64,'Embedding size')
cmd:option('-max_epoch',300,'Max. No. of epochs')
cmd:option('-batch_size',50,'Batch size')
cmd:option('-dump_freq',50,'Models dump freq.')
cmd:option('-delimeter',' ','Separating delimeter, either space or \t')
cmd:text()

-- parse input params
params = cmd:parse(arg)


L1 = params.L1
L2 = params.L2
data_path = params.datadir

lr = params.lr
emb_size = params.embedding_size
lr_decay = params.lr_decay
threshold = params.threshold
max_epoch = params.max_epoch
batch_size = params.batch_size
dump_freq = params.dump_freq
sep = params.delimeter

function printf(s,...)
  return io.write(s:format(...))
end

function printtime(s)
  return string.format("%.2d:%.2d:%.2d", s/(60*60), s/60%60, s%60)
end


function test_model(final,epoch)
  local input1 = split:forward(inputs1[{{1,1000},{}}])
  local input2 = split:forward(inputs2[{{1,1000},{}}])
  local output1 = additive1:forward( input1)
  local output2 = additive2:forward( input2)
  all_roots1 = output1
  all_roots2 = output2

  score = 0
  for idx1 = 1, all_roots1:size(1) do
    closest = idx1
    for idx2 = 1, all_roots2:size(1) do
      if torch.dist(all_roots1[idx1],all_roots2[idx2]) < torch.dist(all_roots1[idx1],all_roots2[closest]) then
        closest = idx2
      end
    end
    
    if idx1 == closest then
      score = score + 1 
    else
      if final == true then print("Closest to: "..idx1.." is: ".. closest) end
    end
  end
  print("Test Score: " .. score.. '/' .. all_roots1:size(1))
  torch.save(L2..'_model'..epoch,additive2)
  torch.save(L2..'_LT'..epoch,lt2)
  torch.save(L1..'_model'..epoch,additive1)
  torch.save(L1..'_LT'..epoch,lt1)
end

corpus_en = Corpus(L1..".all")
corpus_fr = Corpus(L2..".all")
corpus_en.sep = sep
corpus_fr.sep = sep
corpus_en:prepare_corpus()
corpus_fr:prepare_corpus()
no_of_sents = #corpus_en.sequences
inputs1=corpus_en:get_data()
inputs2=corpus_fr:get_data()

torch.save(L1..'_vocab',corpus_en.vocab_map)
torch.save(L2..'_vocab',corpus_fr.vocab_map)

vocab_size1 = corpus_en.no_of_words
vocab_size2 = corpus_fr.no_of_words


additive1 = nn.Sequential()
lt1 = nn.LookupTableMaskZero(vocab_size1,emb_size)
additive1:add( nn.Sequencer(lt1))
additive1:add( nn.CAddTable())
additive1:getParameters():uniform(-0.01,0.01)

additive1:cuda()


additive2 = nn.Sequential()
lt2 = nn.LookupTableMaskZero(vocab_size2,emb_size)
additive2:add( nn.Sequencer(lt2))
additive2:add( nn.CAddTable())
additive2:getParameters():uniform(-0.01,0.01)

additive2:cuda()

criterion = nn.AbsCriterion()
criterion = nn.MaskZeroCriterion(criterion, 1):cuda()
beginning_time = torch.tic()
split = nn.SplitTable(2)
for i =1,max_epoch do
    errors = {}
    local inds = torch.range(1, no_of_sents,batch_size)
    local shuffle = torch.totable(torch.randperm(inds:size(1)))
    for j = 1, no_of_sents/batch_size do 
                --get input row and target
        local start = inds[shuffle[j]]
        local endd = inds[shuffle[j]]+batch_size-1
        local input1 = split:forward(inputs1[{{start,endd},{}}])
        local input2 = split:forward(inputs2[{{start,endd},{}}])
        additive1:zeroGradParameters()
        additive2:zeroGradParameters()
        -- print( target)
        local output1 = additive1:forward( input1)
        local output2 = additive2:forward( input2)
        local err = criterion:forward( output1, output2)
        table.insert( errors, err)
        local gradOutputs = criterion:backward(output1, output2)
        additive1:backward(input1, gradOutputs)
        additive1:updateParameters(lr)
        output1 = additive1:forward( input1)
        output2 = additive2:forward( input2)
        err = criterion:forward( output2, output1)
        table.insert( errors, err)
        gradOutputs = criterion:backward(output2, output1)
        additive2:backward(input2, gradOutputs)
        additive2:updateParameters(lr)
    end
    printf ( 'epoch %4d, loss = %6.50f \n', i, torch.mean( torch.Tensor( errors))   )
    if i % threshold == 0 then lr = lr / lr_decay end
    if i % dump_freq == 0 then test_model(false,i) end
end

durationSeconds = torch.toc(beginning_time)
print('time elapsed:'.. printtime(durationSeconds))
test_model(true,'final')