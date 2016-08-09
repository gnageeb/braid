local stringx = require('pl.stringx')
local file = require('pl.file')

--local data_path = "./data/"
require 'torch'


local Corpus = torch.class("Corpus")

function Corpus:__init(fname)
  self.vocab_idx = 0
  self.vocab_map = {['<pad>'] = 0}
  self.longest_seq = 0
  self.data_file = data_path..fname
  self.x = nil
  self.no_of_words = 0
  self.no_of_sents = 0
  self. sep = ' '
end

function Corpus:padding(sequence)
  -- if extracting word embeddings! 
  new_sequence = {}
  for i = 1 , self.longest_seq - #sequence do
    new_sequence[i] = '<pad>'
  end
  j = 1
  for i = (self.longest_seq - math.min(self.longest_seq,#sequence))+1, self.longest_seq do
    new_sequence[i] = sequence[j]
    j = j + 1
  end
  return new_sequence
end


function Corpus:load_data(fname,max_len)
   local data = file.read(fname)
   -- data = stringx.replace(data, '\n', '<eos>')
   self.sequences = stringx.split(data, '\n')
   all_words = stringx.split(data,self.sep) -- tokenized with tab
   print(string.format("Loading %s, Number of sentences = %d", fname, #self.sequences))
   for i = 1 , #self.sequences do
     local words = stringx.split(self.sequences[i],self.sep) -- tokenized with tab
     if self.longest_seq < table.getn(words) then 
       self.longest_seq = table.getn(words)
     end
   end
   print(string.format("Longest sequence = %d", self.longest_seq))
   if max_len ~= nil then
     self.longest_seq = math.min(self.longest_seq,max_len)
     print(string.format("Reduced down to = %d", self.longest_seq))
   end
   
end

function Corpus:shape_data()
  local x = torch.zeros(#self.sequences, self.longest_seq)
   for i = 1, #self.sequences do
     local words = stringx.split(self.sequences[i], self.sep) -- tokenized with tab
     words = self:padding(words)
     for j = 1, #words do 
       if self.vocab_map[words[j]] == nil then
         self.vocab_idx = self.vocab_idx + 1
         self.vocab_map[words[j]] = self.vocab_idx
       end
       x[i][j] = self.vocab_map[words[j]] 
     end

   end
   print(string.format("Number of words = %d", self.vocab_idx))
   self.no_of_words = self.vocab_idx
   return x
end


function Corpus:prepare_corpus(max_len)
   self:load_data(self.data_file,max_len)
end

function Corpus:get_data()
   self.x = self:shape_data()
   return self.x
end