function printf(s,...)
  return io.write(s:format(...))
end

function printtime(s)
  return string.format("%.2d:%.2d:%.2d", s/(60*60), s/60%60, s%60)
end

function getn(tbl)
  n = 0
  for k in pairs(tbl) do
    n = n + 1
  end 
  return n
end

function flip(tbl)
  a = {}
  for k in pairs(tbl) do
    a[tbl[k]] = k
  end 
  return a
end

function get_labels(vocab)
    flipped = flip(vocab)
    no_of_words = getn(vocab)
    labels = {}
    for i = 0,no_of_words do
      table.insert(labels,flipped[i])
    end
    return labels
end

function subrange(t, first, last)
  local sub = {}
  for i=first,last do
    sub[#sub + 1] = t[i]
  end
  return sub
end

  