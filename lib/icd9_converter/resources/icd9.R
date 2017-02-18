library(icd9)
library(jsonlite)

# Export the data from the R icd9 package as json files.

x = toJSON(ahrqComorbid)
fh = gzfile("ahrqComorbid.json.gz", open="w")
cat(x, file=fh)
close(fh)

x = toJSON(ahrqComorbidAll)
fh = gzfile("ahrqComorbidAll.json.gz", open="w")
cat(x, file=fh)
close(fh)

x = toJSON(elixComorbid)
fh = gzfile("elixComorbid.json.gz", open="w")
cat(x, file=fh)
close(fh)

x = toJSON(icd9Chapters)
fh = gzfile("icd9Chapters.json.gz", open="w")
cat(x, file=fh)
close(fh)

x = toJSON(icd9Hierarchy)
fh = gzfile("icd9Hierarchy.json.gz", open="w")
cat(x, file=fh)
close(fh)

x = toJSON(quanElixComorbid)
fh = gzfile("quanElixComorbid.json.gz", open="w")
cat(x, file=fh)
close(fh)


